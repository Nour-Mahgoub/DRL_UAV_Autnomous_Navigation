"""
Raw-AirSim test that replicates step()'s FULL per-step sequence, not just the
velocity command. raw_pure_forward_test.py showed near-zero z-drift with just
moveByVelocityAsync() called back-to-back — but step() also calls
simGetCollisionInfo() and simGetImages() (depth + segmentation) after every
single velocity command, before issuing the next one.

This script reproduces that exact sequence (velocity command -> collision
check -> two-image capture) using raw AirSim calls, with none of AirSimEnv's
other machinery (no reset() jitter, no segmentation class setup, etc).

If THIS drifts like step_test.py did, the cause is the per-step image-capture/
collision-check overhead sitting between velocity commands (real time elapsing
with no active command), not anything about AirSimEnv itself. If this stays
flat like raw_pure_forward_test.py, the cause is something else in step()/
reset() we haven't isolated yet.

Usage:
    python raw_forward_with_obs_test.py
"""

import airsim
import time
import csv
import os
from datetime import datetime

VX, VY, VZ = 2.0, 0.0, 0.0
DT = 0.5
NUM_STEPS = 20
START_POSITION = (0.0, 0.0, -3.0)


def get_position(client):
    state = client.getMultirotorState()
    p = state.kinematics_estimated.position
    return (p.x_val, p.y_val, p.z_val)


def capture_images(client):
    """Same simGetImages call as AirSimEnv._get_image_features()."""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
    ])
    return responses


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    client.moveToPositionAsync(*START_POSITION, velocity=2.0).join()
    time.sleep(1)

    start = get_position(client)
    print(f"Start position: {start}")
    print(f"Commanding velocity ({VX}, {VY}, {VZ}) for {DT}s, {NUM_STEPS} times, "
          f"WITH collision-check + image-capture between each command (matches step()).\n")

    rows = []
    for i in range(1, NUM_STEPS + 1):
        before = get_position(client)

        t0 = time.time()
        client.moveByVelocityAsync(VX, VY, VZ, DT).join()
        t1 = time.time()

        collision = client.simGetCollisionInfo()  # same as step()
        t2 = time.time()

        capture_images(client)  # same as step()'s _get_observation() -> _get_image_features()
        t3 = time.time()

        after = get_position(client)

        delta = tuple(after[j] - before[j] for j in range(3))
        rows.append({
            "call": i,
            "before_x": before[0], "before_y": before[1], "before_z": before[2],
            "after_x": after[0], "after_y": after[1], "after_z": after[2],
            "delta_x": delta[0], "delta_y": delta[1], "delta_z": delta[2],
            "move_time_s": t1 - t0,
            "collision_check_time_s": t2 - t1,
            "image_capture_time_s": t3 - t2,
        })
        print(f"call {i:2d} | delta=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f}) "
              f"| move={t1-t0:.3f}s collision={t2-t1:.3f}s images={t3-t2:.3f}s")

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"raw_forward_with_obs_test_{run_id}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_dz = rows[-1]["after_z"] - rows[0]["before_z"]
    avg_dz_per_call = total_dz / len(rows)
    avg_image_time = sum(r["image_capture_time_s"] for r in rows) / len(rows)
    print(f"\nTotal z drift over {len(rows)} calls: {total_dz:.3f} m "
          f"(avg {avg_dz_per_call:.4f} m/call)")
    print(f"Average image capture time: {avg_image_time:.3f}s per call")
    print(f"Compare: raw_pure_forward_test.py ~ -0.0037 m/call, step_test.py ~ 0.045-0.047 m/call")
    print(f"log saved to: {log_path}")

    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    main()