# """
# Standalone greedy evaluation of a completed run's saved checkpoint.

# Use this when a run already finished WITHOUT periodic greedy-eval hooks
# active during training (e.g. this Boltzmann run) — loads a saved .pt
# checkpoint and runs a handful of fully-greedy episodes against it, to
# answer: "is the Q-function itself collapsed, or was training success
# rate just being masked/distorted by temperature at the time?"

# Usage:
#     python agents/eval_greedy.py results/<RUN_ID>_ep200.pt --episodes 10
# """

# import argparse
# import numpy as np
# import torch

# from airsim_env import AirSimEnv
# from boltzman_aget import BoltzmannDQNAgent

# # Must match the config the checkpoint was trained with (see the
# # "# ---- config ----" header block in the run's .log file).
# GOAL_POSITION = (20, 0, -3)
# START_POSITION = (0, 0, -3)
# MAX_EPISODE_STEPS = 150
# DT = 0.5
# SPEED = 2.0


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("checkpoint", help="Path to saved .pt checkpoint")
#     parser.add_argument("--episodes", type=int, default=10)
#     args = parser.parse_args()

#     env = AirSimEnv(
#         goal_position=GOAL_POSITION,
#         start_position=START_POSITION,
#         max_episode_steps=MAX_EPISODE_STEPS,
#         dt=DT,
#         speed=SPEED,
#     )

#     agent = BoltzmannDQNAgent(
#         state_dim=AirSimEnv.STATE_DIM,
#         action_dim=env.action_space.n,
#         lr=1e-4,
#         gamma=0.99,
#         buffer_capacity=1_000_000,
#         batch_size=32,
#         temp_start=1.0,
#         temp_end=0.05,
#         temp_decay_steps=24000,
#         target_update_freq=250,
#         use_double=True,
#         use_dueling=True,
#     )
#     agent.q_network.load_state_dict(torch.load(args.checkpoint, map_location=agent.device))
#     agent.q_network.eval()

#     successes, rewards, steps_list = [], [], []
#     for ep in range(1, args.episodes + 1):
#         state = env.reset()
#         done = False
#         info = {}
#         ep_reward = 0.0
#         while not done:
#             action_idx = agent.select_action_greedy(state)
#             state, reward, done, info = env.step(action_idx)
#             ep_reward += reward
#         successes.append(info.get("is_success", False))
#         rewards.append(ep_reward)
#         steps_list.append(info.get("step_num", 0))
#         print(f"Eval ep {ep:2d} | success={info.get('is_success', False)} | "
#               f"reward={ep_reward:7.2f} | steps={info.get('step_num', 0)}")

#     success_rate = 100.0 * sum(successes) / len(successes)
#     print(f"\nCheckpoint: {args.checkpoint}")
#     print(f"Greedy success rate: {success_rate:.1f}% over {args.episodes} episodes")
#     print(f"Avg reward: {np.mean(rewards):.2f}")


# if __name__ == "__main__":
#     main()

"""
Standalone greedy evaluation of a completed run's saved checkpoint.

Use this when a run already finished WITHOUT periodic greedy-eval hooks
active during training (e.g. this Boltzmann run) — loads a saved .pt
checkpoint and runs a handful of fully-greedy episodes against it, to
answer: "is the Q-function itself collapsed, or was training success
rate just being masked/distorted by temperature at the time?"

Usage:
    python agents/eval_greedy.py results/<RUN_ID>_ep200.pt --episodes 10
"""

import argparse
import numpy as np
import torch

from airsim_env import AirSimEnv
from boltzman_aget import BoltzmannDQNAgent  # network/select_action_greedy only — architecture is
                                                # identical to DQNAgent regardless of which class trained it

# Must match the config the checkpoint was trained with (see the
# "# ---- config ----" header block in the run's .log file).
GOAL_POSITION = (70, 30, -5)
START_POSITION = (0, 0, -3)
MAX_EPISODE_STEPS = 150
DT = 0.5
SPEED = 2.0
STATE_DIM = 15   # this run dropped the 5 depth features (20 -> 15)

# TODO: fill these in with the exact values used for this training run —
# guessing wrong here silently biases the eval toward a task that's
# either easier or harder than what the checkpoint was actually trained
# and evaluated on during training.
START_JITTER_RADIUS = 2.0
GOAL_JITTER_RADIUS = 5.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to saved .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    env = AirSimEnv(
        goal_position=GOAL_POSITION,
        start_position=START_POSITION,
        max_episode_steps=MAX_EPISODE_STEPS,
        dt=DT,
        speed=SPEED,
        randomize_positions=(START_JITTER_RADIUS is not None and GOAL_JITTER_RADIUS is not None),
        start_jitter_radius=START_JITTER_RADIUS or 0.0,
        goal_jitter_radius=GOAL_JITTER_RADIUS or 0.0,
    )

    agent = BoltzmannDQNAgent(
        state_dim=STATE_DIM,
        action_dim=env.action_space.n,
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=1_000_000,
        batch_size=32,
        temp_start=1.0,
        temp_end=0.05,
        temp_decay_steps=24000,
        target_update_freq=250,
        use_double=True,
        use_dueling=True,
    )
    agent.q_network.load_state_dict(torch.load(args.checkpoint, map_location=agent.device))
    agent.q_network.eval()

    successes, rewards, steps_list = [], [], []
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        info = {}
        ep_reward = 0.0
        while not done:
            action_idx = agent.select_action_greedy(state)
            state, reward, done, info = env.step(action_idx)
            ep_reward += reward
        successes.append(info.get("is_success", False))
        rewards.append(ep_reward)
        steps_list.append(info.get("step_num", 0))
        end_reason = (
            "success" if info.get("is_success") else
            "crash" if info.get("is_crash") else
            "out_of_bounds" if info.get("is_out_of_bounds") else
            "timeout" if info.get("is_timeout") else
            "unknown"
        )
        print(f"Eval ep {ep:2d} | end={end_reason:12s} | "
              f"reward={ep_reward:7.2f} | steps={info.get('step_num', 0)} | "
              f"final_dist={info.get('dist_to_goal', float('nan')):.2f}")

    success_rate = 100.0 * sum(successes) / len(successes)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Greedy success rate: {success_rate:.1f}% over {args.episodes} episodes")
    print(f"Avg reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()