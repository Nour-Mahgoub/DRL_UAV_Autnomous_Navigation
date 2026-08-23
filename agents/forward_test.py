"""
Raw-AirSim version of step_only_test.py — apples-to-apples comparison.

Commands the exact same thing as step_test.py's forward action: constant
velocity (2.0, 0.0, 0.0) for dt=0.5s, repeated 18-20 times, with NO
goal-tracking or z-correction of any kind. This bypasses AirSimEnv/step()
completely, so it isolates whether the small steady z-drift seen in
step_test.py is:

  (a) inherent AirSim/quadrotor physics (forward pitch -> slight altitude
      sag under sustained acceleration) — expected, not a bug, and this
      script will show a similar drift too, or

  (b) something specific to step()/AirSimEnv — in which case this script
      should stay flat while step_test.py drifts.

Usage:
    python raw_pure_forward_test.py
"""

import airsim
import time
import csv
import os
from datetime import datetime

VX, VY, VZ = 2.0, 0.0, 0.0   # same as FORWARD_ACTION=22 decodes to
DT = 0.5
NUM_STEPS = 20
START_POSITION = (0.0, 0.0, -3.0)


def get_position(client):
    state = client.getMultirotorState()
    p = state.kinematics_estimated.position
    return (p.x_val, p.y_val, p.z_val)


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
    print(f"Commanding constant velocity ({VX}, {VY}, {VZ}) for {DT}s, {NUM_STEPS} times, no correction.\n")

    rows = []
    for i in range(1, NUM_STEPS + 1):
        before = get_position(client)
        client.moveByVelocityAsync(VX, VY, VZ, DT).join()
        after = get_position(client)

        delta = tuple(after[j] - before[j] for j in range(3))
        rows.append({
            "call": i,
            "before_x": before[0], "before_y": before[1], "before_z": before[2],
            "after_x": after[0], "after_y": after[1], "after_z": after[2],
            "delta_x": delta[0], "delta_y": delta[1], "delta_z": delta[2],
        })
        print(f"call {i:2d} | before=({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f}) "
              f"-> after=({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f}) "
              f"| delta=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f})")

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"raw_pure_forward_test_{run_id}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_dz = rows[-1]["after_z"] - rows[0]["before_z"]
    avg_dz_per_call = total_dz / len(rows)
    print(f"\nTotal z drift over {len(rows)} calls: {total_dz:.3f} m "
          f"(avg {avg_dz_per_call:.4f} m/call)")
    print(f"Compare against step_test.py's ~0.045-0.047 m/call drift.")
    print(f"log saved to: {log_path}")

    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    main()