# import time
# import math
# import numpy as np
# import airsim

# from action_space import ActionSpace
# from reward import compute_reward


# class AirSimEnv:
#     """
#     Gym-style environment wrapping AirSim for the Blocks map.
#     Interface matches gymnasium: reset() -> obs, step(action) -> (obs, reward, done, info)

#     Observation vector (20 dims):
#         position (3): x, y, z
#         velocity (3): vx, vy, vz
#         yaw (1): current heading, radians
#         relative goal vector (3): dx, dy, dz to goal
#         depth obstacle features (5): min depth per horizontal zone, normalized
#         segmentation obstacle features (5): obstacle pixel density per zone,
#             using explicit semantic classes (ground=0, cylinder=1, cube=2,
#             cone=3, orange ball=4, assigned via simSetSegmentationObjectID
#             at init) — fraction of pixels in each zone belonging to any
#             real obstacle class, not ground/background. Genuinely semantic,
#             not a raw-color proxy, since classes are deliberately assigned
#             based on Blocks' actual mesh names.
#     """

#     STATE_DIM = 20

#     def __init__(self, goal_position, start_position=(0, 0, -3),
#                  max_episode_steps=300, dt=0.5, speed=2.0,
#                  accept_radius=2.0, max_depth_meters=20.0,
#                  workspace_bounds=None):
#         self.client = airsim.MultirotorClient()
#         self.client.confirmConnection()
#         self.client.enableApiControl(True)
#         self.client.armDisarm(True)

#         self._configure_segmentation_classes()

#         self.action_space = ActionSpace(speed=speed)
#         self.dt = dt

#         self.start_position = np.array(start_position, dtype=np.float32)
#         self.goal_position = np.array(goal_position, dtype=np.float32)
#         self.max_episode_steps = max_episode_steps
#         self.accept_radius = accept_radius
#         self.max_depth_meters = max_depth_meters

#         # workspace bounds: dict with x/y/z min-max, used to end episode if drone flies out of bounds
#         self.workspace_bounds = workspace_bounds or {
#             "x": (-200, 200), "y": (-200, 200), "z": (-100, 0)
#         }

#         self.step_num = 0
#         self.prev_dist = None

#     # ---------------- reset ----------------

#     def _configure_segmentation_classes(self):
#         """
#         Assigns explicit semantic segmentation IDs based on Blocks' actual
#         mesh names (confirmed via simListSceneObjects()):
#             class 0: Ground, Ground_2..Ground_6      -> traversable/background
#             class 1: Cylinder2..Cylinder8, Cylinder_2 -> cylindrical obstacles
#             class 2: TemplateCube_Rounded_*           -> cube obstacles (main obstacle field)
#             class 3: Cone_5                           -> cone obstacle (separate from cylinder)
#             class 4: OrangeBall                       -> spherical obstacle

#         Everything else (cameras, lights, sky, UI actors) is left at its
#         default ID — non-physical scene elements, not real obstacles.

#         Called once at env init so every episode uses the same mapping.
#         With classes explicitly assigned, "differs from the frame's
#         dominant color" (used in _extract_segmentation_features) reliably
#         means "is a real obstacle", since Ground is almost always the
#         dominant color in an open scene — this makes the density
#         calculation genuinely semantic rather than an arbitrary
#         per-mesh-color heuristic.
#         """
#         self.client.simSetSegmentationObjectID("Ground.*", 0, True)
#         self.client.simSetSegmentationObjectID("Cylinder.*", 1, True)
#         self.client.simSetSegmentationObjectID("TemplateCube_Rounded.*", 2, True)
#         self.client.simSetSegmentationObjectID("Cone.*", 3, True)
#         self.client.simSetSegmentationObjectID("OrangeBall", 4, True)

#     def reset(self):
#         self.client.reset()
#         self.client.enableApiControl(True)
#         self.client.armDisarm(True)

#         self.client.takeoffAsync().join()
#         self._move_to_start()

#         # wait for collision flag to clear before starting the episode —
#         # spawn/landing contact registers as a transient collision (confirmed in smoke test)
#         self._wait_for_clean_state()

#         self.step_num = 0
#         self.prev_dist = self._distance_to_goal()

#         obs = self._get_observation()
#         return obs

#     def _move_to_start(self):
#         self.client.moveToPositionAsync(
#             float(self.start_position[0]),
#             float(self.start_position[1]),
#             float(self.start_position[2]),
#             velocity=2.0,
#         ).join()

#     def _wait_for_clean_state(self, timeout=3.0, poll_interval=0.1):
#         elapsed = 0.0
#         while self.client.simGetCollisionInfo().has_collided and elapsed < timeout:
#             time.sleep(poll_interval)
#             elapsed += poll_interval
#         # if still colliding after timeout, proceed anyway and log it —
#         # don't hang the training loop indefinitely on a stuck flag
#         if self.client.simGetCollisionInfo().has_collided:
#             print(f"Warning: collision flag still True after {timeout}s wait at reset.")

#     # ---------------- step ----------------

#     def step(self, action_idx):
#         vx, vy, vz = self.action_space.decode(action_idx)

#         self.client.moveByVelocityAsync(
#             vx, vy, vz, self.dt,
#             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
#             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
#         ).join()

#         curr_dist = self._distance_to_goal()
#         collided = self.client.simGetCollisionInfo().has_collided

#         reward = compute_reward(
#             prev_dist=self.prev_dist,
#             curr_dist=curr_dist,
#             collided=collided,
#             action_vector=[vx, vy, vz],
#         )
#         self.prev_dist = curr_dist

#         self.step_num += 1
#         reached_goal = curr_dist < self.accept_radius
#         out_of_bounds = self._is_out_of_bounds()
#         timed_out = self.step_num >= self.max_episode_steps

#         done = collided or reached_goal or out_of_bounds or timed_out

#         info = {
#             "is_success": reached_goal,
#             "is_crash": collided,
#             "is_out_of_bounds": out_of_bounds,
#             "is_timeout": timed_out,
#             "step_num": self.step_num,
#             "dist_to_goal": curr_dist,
#         }

#         obs = self._get_observation()
#         return obs, reward, done, info

#     # ---------------- observation ----------------

#     def _get_observation(self):
#         state = self.client.getMultirotorState()
#         pos = state.kinematics_estimated.position
#         vel = state.kinematics_estimated.linear_velocity
#         orientation = state.kinematics_estimated.orientation

#         position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
#         velocity = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)

#         _, _, yaw = airsim.to_eularian_angles(orientation)
#         yaw = np.array([yaw], dtype=np.float32)

#         relative_goal = (self.goal_position - position).astype(np.float32)

#         obstacle_features, segmentation_features = self._get_image_features()

#         obs = np.concatenate([position, velocity, yaw, relative_goal,
#                                obstacle_features, segmentation_features])
#         assert obs.shape[0] == self.STATE_DIM, f"Expected {self.STATE_DIM}-dim obs, got {obs.shape[0]}"
#         return obs

#     def _get_image_features(self, num_zones=5):
#         """
#         Single simGetImages call requesting both depth and segmentation,
#         to avoid two separate round trips to AirSim per step.
#         Returns (depth_features, segmentation_features), each shape (num_zones,).
#         """
#         responses = self.client.simGetImages([
#             airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
#             airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
#         ])

#         if not responses or len(responses) < 2:
#             return (np.ones(num_zones, dtype=np.float32),
#                     np.zeros(num_zones, dtype=np.float32))

#         depth_features = self._extract_depth_features(responses[0], num_zones)
#         segmentation_features = self._extract_segmentation_features(responses[1], num_zones)

#         return depth_features, segmentation_features

#     def _extract_depth_features(self, response, num_zones=5):
#         """
#         Splits the depth image into `num_zones` horizontal zones,
#         returns the minimum depth per zone (closest obstacle), normalized to [0, 1].
#         1.0 = no obstacle within max_depth_meters, 0.0 = obstacle right at the drone.
#         """
#         if response.width == 0:
#             return np.ones(num_zones, dtype=np.float32)

#         depth_img = airsim.list_to_2d_float_array(
#             response.image_data_float,
#             response.width,
#             response.height,
#         )

#         zone_splits = np.array_split(depth_img, num_zones, axis=1)
#         min_depths = np.array([zone.min() for zone in zone_splits], dtype=np.float32)

#         clipped = np.clip(min_depths, 0, self.max_depth_meters)
#         normalized = clipped / self.max_depth_meters
#         return normalized

#     def _extract_segmentation_features(self, response, num_zones=5):
#         """
#         Splits the segmentation image into `num_zones` horizontal zones,
#         returns the fraction of "non-ground" pixels per zone — i.e. pixels
#         belonging to explicit obstacle classes (cylinder=1, cube=2), not
#         ground/background (class 0). Since ground is almost always the
#         dominant color in an open scene, "differs from the frame's
#         dominant color" reliably identifies real obstacle classes here,
#         because classes were explicitly assigned in _configure_segmentation_classes()
#         rather than left as arbitrary per-mesh colors.
#         """
#         if response.width == 0:
#             return np.zeros(num_zones, dtype=np.float32)

#         img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
#         try:
#             img_rgb = img1d.reshape(response.height, response.width, 3)
#         except ValueError:
#             # unexpected buffer size — fail safe rather than crash the training loop
#             return np.zeros(num_zones, dtype=np.float32)

#         # estimate background (ground) color as the most frequent pixel color in the full frame
#         flat = img_rgb.reshape(-1, 3)
#         colors, counts = np.unique(flat, axis=0, return_counts=True)
#         background_color = colors[np.argmax(counts)]

#         zone_splits = np.array_split(img_rgb, num_zones, axis=1)
#         features = []
#         for zone in zone_splits:
#             zone_flat = zone.reshape(-1, 3)
#             non_background = np.any(zone_flat != background_color, axis=1)
#             obstacle_fraction = float(non_background.mean())
#             features.append(obstacle_fraction)

#         return np.array(features, dtype=np.float32)

#     # ---------------- helpers ----------------

#     def _distance_to_goal(self):
#         state = self.client.getMultirotorState()
#         pos = state.kinematics_estimated.position
#         current = np.array([pos.x_val, pos.y_val, pos.z_val])
#         return float(np.linalg.norm(self.goal_position - current))

#     def _is_out_of_bounds(self):
#         state = self.client.getMultirotorState()
#         pos = state.kinematics_estimated.position
#         x_ok = self.workspace_bounds["x"][0] <= pos.x_val <= self.workspace_bounds["x"][1]
#         y_ok = self.workspace_bounds["y"][0] <= pos.y_val <= self.workspace_bounds["y"][1]
#         z_ok = self.workspace_bounds["z"][0] <= pos.z_val <= self.workspace_bounds["z"][1]
#         return not (x_ok and y_ok and z_ok)

import time
import math
import random
import numpy as np
import airsim

from action_space import ActionSpace
from reward import compute_reward


class AirSimEnv:
    """
    Gym-style environment wrapping AirSim for the Blocks map.
    Interface matches gymnasium: reset() -> obs, step(action) -> (obs, reward, done, info)

    Observation vector (20 dims):
        position (3): x, y, z
        velocity (3): vx, vy, vz
        yaw (1): current heading, radians
        relative goal vector (3): dx, dy, dz to goal
        depth obstacle features (5): min depth per horizontal zone, normalized
        segmentation obstacle features (5): obstacle pixel density per zone,
            using explicit semantic classes (ground=0, cylinder=1, cube=2,
            cone=3, orange ball=4, assigned via simSetSegmentationObjectID
            at init) — fraction of pixels in each zone belonging to any
            real obstacle class, not ground/background. Genuinely semantic,
            not a raw-color proxy, since classes are deliberately assigned
            based on Blocks' actual mesh names.

    START/GOAL RANDOMIZATION (added to break single-trajectory memorization —
    a fixed start+goal every episode lets the policy converge on one open-loop
    action sequence instead of learning to condition on the observation):

        scenario_pairs: optional list of (start, goal) tuples. If provided,
            reset() picks one at random each episode. Use this once you've
            manually verified a handful of safe (start, goal) pairs — the
            most robust option, but requires that upfront curation.

        randomize_positions + start_jitter_radius / goal_jitter_radius:
            if scenario_pairs is NOT provided and randomize_positions=True,
            reset() adds a random offset (uniform within a sphere of the
            given radius, in meters) around the base start_position /
            goal_position each episode. Lower-effort, lower-risk than
            curating new pairs, since it perturbs coordinates you've
            already confirmed are reachable rather than introducing new
            ones that might land inside an obstacle mesh.

        Both default to off (jitter radius 0, no scenario_pairs), so
        existing single-pair training scripts behave exactly as before
        unless you opt in.
    """

    STATE_DIM = 20

    def __init__(self, goal_position, start_position=(0, 0, -3),
                 max_episode_steps=300, dt=0.5, speed=2.0,
                 accept_radius=2.0, max_depth_meters=20.0,
                 workspace_bounds=None,
                 scenario_pairs=None,
                 randomize_positions=True,
                 start_jitter_radius=1.0,
                 goal_jitter_radius=3.0,
                 max_reset_attempts=5):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self._configure_segmentation_classes()

        self.action_space = ActionSpace(speed=speed)
        self.dt = dt

        # base/reference positions — used directly if no randomization is
        # configured, or as the center of the jitter sphere if it is
        self.base_start_position = np.array(start_position, dtype=np.float32)
        self.base_goal_position = np.array(goal_position, dtype=np.float32)

        # the ACTIVE positions used for this episode — reset() sets these
        # fresh every call when randomization is on
        self.start_position = self.base_start_position.copy()
        self.goal_position = self.base_goal_position.copy()

        self.scenario_pairs = scenario_pairs
        self.randomize_positions = randomize_positions
        self.start_jitter_radius = start_jitter_radius
        self.goal_jitter_radius = goal_jitter_radius
        self.max_reset_attempts = max_reset_attempts

        self.max_episode_steps = max_episode_steps
        self.accept_radius = accept_radius
        self.max_depth_meters = max_depth_meters

        # workspace bounds: dict with x/y/z min-max, used to end episode if drone flies out of bounds
        self.workspace_bounds = workspace_bounds or {
            "x": (-100, 100), "y": (-100, 100), "z": (-50, 0)
        }

        self.step_num = 0
        self.prev_dist = None

    # ---------------- reset ----------------

    def _configure_segmentation_classes(self):
        """
        Assigns explicit semantic segmentation IDs based on Blocks' actual
        mesh names (confirmed via simListSceneObjects()):
            class 0: Ground, Ground_2..Ground_6      -> traversable/background
            class 1: Cylinder2..Cylinder8, Cylinder_2 -> cylindrical obstacles
            class 2: TemplateCube_Rounded_*           -> cube obstacles (main obstacle field)
            class 3: Cone_5                           -> cone obstacle (separate from cylinder)
            class 4: OrangeBall                       -> spherical obstacle

        Everything else (cameras, lights, sky, UI actors) is left at its
        default ID — non-physical scene elements, not real obstacles.

        Called once at env init so every episode uses the same mapping.
        """
        self.client.simSetSegmentationObjectID("Ground.*", 0, True)
        self.client.simSetSegmentationObjectID("Cylinder.*", 1, True)
        self.client.simSetSegmentationObjectID("TemplateCube_Rounded.*", 2, True)
        self.client.simSetSegmentationObjectID("Cone.*", 3, True)
        self.client.simSetSegmentationObjectID("OrangeBall", 4, True)

    def _sample_jittered_position(self, base, radius):
        """
        Uniform-random offset within a sphere of `radius` meters around
        `base`. Result is clipped to workspace_bounds (with a small
        safety margin) so jitter can never sample outside the box your
        own out-of-bounds check enforces.
        """
        if radius <= 0:
            sampled = base.copy()
        else:
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-8
            # cube-root scaling -> uniform density within the sphere's
            # volume, not just uniform radius (which would bias samples
            # toward the surface)
            mag = radius * (np.random.uniform() ** (1.0 / 3.0))
            sampled = base + direction * mag

        margin = 1.0
        for i, axis in enumerate(("x", "y", "z")):
            lo, hi = self.workspace_bounds[axis]
            sampled[i] = np.clip(sampled[i], lo + margin, hi - margin)

        return sampled.astype(np.float32)

    def _sample_episode_positions(self):
        """
        Chooses this episode's (start, goal). Priority:
          1. scenario_pairs, if provided — a curated, pre-verified set.
          2. randomize_positions jitter around the base pair.
          3. fall back to the fixed base pair (original behavior).
        """
        if self.scenario_pairs:
            start, goal = random.choice(self.scenario_pairs)
            return np.array(start, dtype=np.float32), np.array(goal, dtype=np.float32)

        if self.randomize_positions:
            start = self._sample_jittered_position(self.base_start_position, self.start_jitter_radius)
            goal = self._sample_jittered_position(self.base_goal_position, self.goal_jitter_radius)
            return start, goal

        return self.base_start_position.copy(), self.base_goal_position.copy()

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        for attempt in range(1, self.max_reset_attempts + 1):
            self.start_position, self.goal_position = self._sample_episode_positions()
            self._move_to_start()
            self._wait_for_clean_state()

            if not self.client.simGetCollisionInfo().has_collided:
                break
            print(f"Reset attempt {attempt}/{self.max_reset_attempts}: "
                  f"start position collided, resampling...")
        else:
            print("Warning: could not find a collision-free start after "
                  f"{self.max_reset_attempts} attempts; proceeding anyway.")

        self.step_num = 0
        self.prev_dist = self._distance_to_goal()

        obs = self._get_observation()
        return obs

    def _move_to_start(self):
        self.client.moveToPositionAsync(
            float(self.start_position[0]),
            float(self.start_position[1]),
            float(self.start_position[2]),
            velocity=2.0,
        ).join()

    def _wait_for_clean_state(self, timeout=3.0, poll_interval=0.1):
        elapsed = 0.0
        while self.client.simGetCollisionInfo().has_collided and elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
        # if still colliding after timeout, proceed anyway and log it —
        # don't hang the training loop indefinitely on a stuck flag
        if self.client.simGetCollisionInfo().has_collided:
            print(f"Warning: collision flag still True after {timeout}s wait at reset.")

    # ---------------- step ----------------

    def step(self, action_idx):
        vx, vy, vz = self.action_space.decode(action_idx)

        self.client.moveByVelocityAsync(
            vx, vy, vz, self.dt,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
        ).join()

        curr_dist = self._distance_to_goal()
        collided = self.client.simGetCollisionInfo().has_collided

        reward = compute_reward(
            prev_dist=self.prev_dist,
            curr_dist=curr_dist,
            collided=collided,
            action_vector=[vx, vy, vz],
        )
        self.prev_dist = curr_dist

        self.step_num += 1
        reached_goal = curr_dist < self.accept_radius
        out_of_bounds = self._is_out_of_bounds()
        timed_out = self.step_num >= self.max_episode_steps

        done = collided or reached_goal or out_of_bounds or timed_out

        info = {
            "is_success": reached_goal,
            "is_crash": collided,
            "is_out_of_bounds": out_of_bounds,
            "is_timeout": timed_out,
            "step_num": self.step_num,
            "dist_to_goal": curr_dist,
        }

        obs = self._get_observation()
        return obs, reward, done, info

    # ---------------- observation ----------------

    def _get_observation(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation

        position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)

        _, _, yaw = airsim.to_eularian_angles(orientation)
        yaw = np.array([yaw], dtype=np.float32)

        relative_goal = (self.goal_position - position).astype(np.float32)

        obstacle_features, segmentation_features = self._get_image_features()

        obs = np.concatenate([position, velocity, yaw, relative_goal, obstacle_features, 
                                segmentation_features])
        assert obs.shape[0] == self.STATE_DIM, f"Expected {self.STATE_DIM}-dim obs, got {obs.shape[0]}"
        return obs

    def _get_image_features(self, num_zones=5):
        """
        Single simGetImages call requesting both depth and segmentation,
        to avoid two separate round trips to AirSim per step.
        Returns (depth_features, segmentation_features), each shape (num_zones,).
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
        ])

        if not responses or len(responses) < 2:
            return (np.ones(num_zones, dtype=np.float32),
                    np.zeros(num_zones, dtype=np.float32))

        depth_features = self._extract_depth_features(responses[0], num_zones)
        segmentation_features = self._extract_segmentation_features(responses[1], num_zones)

        return depth_features, segmentation_features

    def _extract_depth_features(self, response, num_zones=5):
        """
        Splits the depth image into `num_zones` horizontal zones,
        returns the minimum depth per zone (closest obstacle), normalized to [0, 1].
        1.0 = no obstacle within max_depth_meters, 0.0 = obstacle right at the drone.
        """
        if response.width == 0:
            return np.ones(num_zones, dtype=np.float32)

        depth_img = airsim.list_to_2d_float_array(
            response.image_data_float,
            response.width,
            response.height,
        )

        zone_splits = np.array_split(depth_img, num_zones, axis=1)
        min_depths = np.array([zone.min() for zone in zone_splits], dtype=np.float32)

        clipped = np.clip(min_depths, 0, self.max_depth_meters)
        normalized = clipped / self.max_depth_meters
        return normalized

    def _extract_segmentation_features(self, response, num_zones=5):
        """
        Splits the segmentation image into `num_zones` horizontal zones,
        returns the fraction of "non-ground" pixels per zone — i.e. pixels
        belonging to explicit obstacle classes (cylinder=1, cube=2), not
        ground/background (class 0).
        """
        if response.width == 0:
            return np.zeros(num_zones, dtype=np.float32)

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        try:
            img_rgb = img1d.reshape(response.height, response.width, 3)
        except ValueError:
            # unexpected buffer size — fail safe rather than crash the training loop
            return np.zeros(num_zones, dtype=np.float32)

        # estimate background (ground) color as the most frequent pixel color in the full frame
        flat = img_rgb.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        background_color = colors[np.argmax(counts)]

        zone_splits = np.array_split(img_rgb, num_zones, axis=1)
        features = []
        for zone in zone_splits:
            zone_flat = zone.reshape(-1, 3)
            non_background = np.any(zone_flat != background_color, axis=1)
            obstacle_fraction = float(non_background.mean())
            features.append(obstacle_fraction)

        return np.array(features, dtype=np.float32)

    # ---------------- helpers ----------------

    def _distance_to_goal(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current = np.array([pos.x_val, pos.y_val, pos.z_val])
        return float(np.linalg.norm(self.goal_position - current))

    def _is_out_of_bounds(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        x_ok = self.workspace_bounds["x"][0] <= pos.x_val <= self.workspace_bounds["x"][1]
        y_ok = self.workspace_bounds["y"][0] <= pos.y_val <= self.workspace_bounds["y"][1]
        z_ok = self.workspace_bounds["z"][0] <= pos.z_val <= self.workspace_bounds["z"][1]
        return not (x_ok and y_ok and z_ok)