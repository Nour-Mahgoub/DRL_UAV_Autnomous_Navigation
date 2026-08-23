"""
Minimal step()-only test.

Purpose: isolate the env wrapper's step() function itself, nothing else. No raw
AirSim calls of our own — everything goes through AirSimEnv.step(). Resets once,
then calls step() with a single fixed "forward" action 10 times in a row, logging
the drone's position immediately before and immediately after each call.

This answers one narrow question: does step(forward_action) reliably move the
drone forward by roughly what you'd expect (speed * dt per call), or does
something break inside step() itself (action decoding, the moveByVelocityAsync
call, or the observation/position readback)?

Usage:
    python step_only_test.py
"""

import csv
import os
from datetime import datetime

from airsim_env import AirSimEnv

START_POSITION = (0.0, 0.0, -3.0)
GOAL_POSITION = (20.0, 0.0, -3.0)   # goal only matters for env init; not used for action choice here
NUM_STEPS = 20

# Hardcode the action index that corresponds to "forward" (+x velocity, vy=0, vz=0)
# in YOUR action_space.py encoding. action_table = itertools.product([-1,0,1] x3),
# so index = x_idx*9 + y_idx*3 + z_idx with x_idx/y_idx/z_idx in {0,1,2} for {-1,0,1}.
# (1, 0, 0) -> x_idx=2, y_idx=1, z_idx=1 -> 2*9 + 1*3 + 1 = 22.
FORWARD_ACTION = 22


def get_position(env):
    state = env.client.getMultirotorState()
    p = state.kinematics_estimated.position
    return (p.x_val, p.y_val, p.z_val)


def main():
    env = AirSimEnv(
        goal_position=GOAL_POSITION,
        start_position=START_POSITION,
        max_episode_steps=NUM_STEPS + 5,  # just needs to not time out mid-test
        randomize_positions=False,
    )

    obs = env.reset()

    vx, vy, vz = env.action_space.decode(FORWARD_ACTION)
    print(f"Using action {FORWARD_ACTION} as 'forward': decodes to velocity ({vx:.2f}, {vy:.2f}, {vz:.2f})")
    print("If that's not (speed, 0, 0), update FORWARD_ACTION at the top of this file.\n")

    print(f"Start position after reset(): {get_position(env)}")
    print(f"Calling step({FORWARD_ACTION}) {NUM_STEPS} times, logging before/after position each call.\n")

    rows = []
    for i in range(1, NUM_STEPS + 1):
        before = get_position(env)
        obs, reward, done, info = env.step(FORWARD_ACTION)
        after = get_position(env)

        delta = tuple(after[j] - before[j] for j in range(3))

        row = {
            "call": i,
            "before_x": before[0], "before_y": before[1], "before_z": before[2],
            "after_x": after[0], "after_y": after[1], "after_z": after[2],
            "delta_x": delta[0], "delta_y": delta[1], "delta_z": delta[2],
            "reward": reward,
            "done": done,
            "is_crash": info["is_crash"],
            "is_out_of_bounds": info["is_out_of_bounds"],
        }
        rows.append(row)

        print(f"call {i:2d} | before=({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f}) "
              f"-> after=({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f}) "
              f"| delta=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f}) "
              f"| reward={reward:.3f} | done={done}")

        if done:
            print(f"  -> episode ended early (crash={info['is_crash']}, oob={info['is_out_of_bounds']})")
            break

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"step_only_test_{run_id}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_dx = rows[-1]["after_x"] - rows[0]["before_x"]
    print(f"\nTotal x displacement over {len(rows)} calls: {total_dx:.3f} m "
          f"(expected roughly {env.dt * 2.0 * len(rows):.3f} m if speed=2.0)")
    print(f"log saved to: {log_path}")


if __name__ == "__main__":
    main()