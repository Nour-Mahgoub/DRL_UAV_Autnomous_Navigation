"""
Standalone test for AirSimEnv — run this BEFORE train_airsim.py.

Checks:
1. reset() returns a 15-dim observation without error
2. step() with random actions works and returns sane values
3. obstacle features respond to flying toward a wall (manual visual check)

Run with the AirSim Blocks simulation already in Play mode.
"""

import numpy as np
from airsim_env import AirSimEnv

# adjust this to an actual reachable point in your Blocks environment
GOAL_POSITION = (20, 0, -3)


def main():
    env = AirSimEnv(
        goal_position=GOAL_POSITION,
        start_position=(0, 0, -3),
        max_episode_steps=20,   # short for testing
        dt=0.5,
        speed=2.0,
    )

    print("Calling reset()...")
    obs = env.reset()
    print(f"Observation shape: {obs.shape} (expect (20,))")
    print(f"Observation values: {obs}")
    print(f"  depth zones (idx 10-14): {np.round(obs[10:15], 2)}")
    print(f"  segmentation zones (idx 15-19): {np.round(obs[15:20], 2)}")
    print()

    print("Running 10 random-action steps...")
    for i in range(10):
        action_idx = np.random.randint(0, env.action_space.n)
        label = env.action_space.describe(action_idx)
        obs, reward, done, info = env.step(action_idx)

        print(f"Step {i+1:2d} | action={action_idx:2d} ({label:20s}) | "
              f"reward={reward:7.3f} | dist_to_goal={info['dist_to_goal']:6.2f} | "
              f"done={done} | depth={np.round(obs[10:15], 2)} | "
              f"seg={np.round(obs[15:20], 2)}")

        if done:
            print(f"Episode ended early: {info}")
            break

    print("\nTest complete. Check that:")
    print("- Observation shape was (15,) with no crash")
    print("- Position/velocity/yaw values looked physically reasonable")
    print("- Obstacle zone values were between 0 and 1")
    print("- Reward values were finite numbers, not NaN/inf")


if __name__ == "__main__":
    main()