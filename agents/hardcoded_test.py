"""
Hardcoded flight test — no RL policy involved.

Purpose: isolate whether the post-migration failures come from the AirSim/Unreal
environment itself (physics, coordinate frame, action execution) rather than from
the RL policy, reward function, or hyperparameters.

This script commands the drone directly toward the goal every step, using the same
step size (dt, speed) your training env uses, but with NO agent, NO epsilon, NO
model checkpoint. If this fails to reach the goal, the problem is environment-side.
If it succeeds, the problem is more likely in airsim_env.py's action mapping,
observation construction, or reward logic.

Usage:
    python hardcoded_flight_test.py
"""

import airsim
import time
import math
import csv
import os
from datetime import datetime

# ---- config (matches your logged run configs) ----
START_POSITION = (0.0, 0.0, -3.0)
GOAL_POSITION = (20.0, 0.0, -3.0)
DT = 0.5                # seconds per step, matches your env
SPEED = 2.0              # m/s, matches your env's action magnitude
MAX_STEPS = 150           # matches max_episode_steps
SUCCESS_RADIUS = 2.0      # matches accept_radius in airsim_env.py
OOB_BOUNDS = {            # matches default workspace_bounds in airsim_env.py
    "x_min": -100, "x_max": 100,
    "y_min": -100, "y_max": 100,
    "z_min": -50, "z_max": 0,
}
# ----------------------------------------------------


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def is_out_of_bounds(pos):
    x, y, z = pos
    return not (
        OOB_BOUNDS["x_min"] <= x <= OOB_BOUNDS["x_max"]
        and OOB_BOUNDS["y_min"] <= y <= OOB_BOUNDS["y_max"]
        and OOB_BOUNDS["z_min"] <= z <= OOB_BOUNDS["z_max"]
    )


def get_position(client):
    state = client.getMultirotorState()
    p = state.kinematics_estimated.position
    return (p.x_val, p.y_val, p.z_val)


def get_velocity(client):
    state = client.getMultirotorState()
    v = state.kinematics_estimated.linear_velocity
    return (v.x_val, v.y_val, v.z_val)


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()

    print(f"Moving to start position {START_POSITION}...")
    client.moveToPositionAsync(*START_POSITION, velocity=SPEED).join()
    time.sleep(1)

    pos = get_position(client)
    print(f"At start: {pos} (target was {START_POSITION})")

    # AirSim's has_collided flag reflects the LAST collision, which can be
    # stale (e.g. from takeoff or the move-to-start command) rather than a
    # real crash during this flight. Record the collision timestamp now as
    # a baseline, so we only count a NEW collision (newer timestamp) as a
    # real crash during the loop below.
    baseline_collision = client.simGetCollisionInfo()
    baseline_collision_time = baseline_collision.time_stamp
    print(f"Baseline collision timestamp: {baseline_collision_time} "
          f"(has_collided={baseline_collision.has_collided} — ignored if stale)")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"hardcoded_flight_log_{run_id}.csv")

    rows = []
    is_success = False
    is_crash = False
    is_oob = False
    is_timeout = False

    print(f"Flying straight toward goal {GOAL_POSITION}, step by step (dt={DT}, speed={SPEED})...")

    for step in range(1, MAX_STEPS + 1):
        pos = get_position(client)
        vel = get_velocity(client)
        d_to_goal = dist(pos, GOAL_POSITION)

        collision = client.simGetCollisionInfo()
        if collision.has_collided and collision.time_stamp > baseline_collision_time:
            is_crash = True

        oob = is_out_of_bounds(pos)
        if oob:
            is_oob = True

        rows.append({
            "step": step,
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
            "dist_to_goal": d_to_goal,
            "has_collided": collision.has_collided,
            "is_out_of_bounds": oob,
        })

        print(f"  step {step:3d} | pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
              f"| vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) | dist_to_goal={d_to_goal:.2f}")

        if d_to_goal <= SUCCESS_RADIUS:
            is_success = True
            print(f"SUCCESS at step {step}: reached goal (dist={d_to_goal:.2f})")
            break

        if is_crash:
            print(f"CRASH at step {step}: collision detected")
            break

        if is_oob:
            print(f"OUT OF BOUNDS at step {step}: pos={pos}")
            break

        # command velocity straight toward goal, magnitude = SPEED, held for DT seconds
        direction = tuple(GOAL_POSITION[i] - pos[i] for i in range(3))
        norm = math.sqrt(sum(c ** 2 for c in direction))
        if norm > 1e-6:
            unit = tuple(c / norm for c in direction)
        else:
            unit = (0.0, 0.0, 0.0)
        vx, vy, vz = (unit[0] * SPEED, unit[1] * SPEED, unit[2] * SPEED)

        client.moveByVelocityAsync(vx, vy, vz, DT).join()

    else:
        is_timeout = True
        print(f"TIMEOUT: did not reach goal within {MAX_STEPS} steps")

    # write CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n---- SUMMARY ----")
    print(f"is_success: {is_success}")
    print(f"is_crash: {is_crash}")
    print(f"is_out_of_bounds: {is_oob}")
    print(f"is_timeout: {is_timeout}")
    print(f"steps taken: {len(rows)}")
    print(f"final position: {rows[-1]['pos_x']:.2f}, {rows[-1]['pos_y']:.2f}, {rows[-1]['pos_z']:.2f}")
    print(f"final distance to goal: {rows[-1]['dist_to_goal']:.2f}")
    print(f"log saved to: {log_path}")

    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    main()