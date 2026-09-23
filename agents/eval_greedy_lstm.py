import argparse

import numpy as np
import torch

from airsim_env import AirSimEnv
from d3rqn_agent import D3RQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to saved .pt checkpoint (q_network state_dict)")
    parser.add_argument("--env-name", default="blocks",
                         help="Must match the configs/<env_name>.yaml this checkpoint was trained on.")
    parser.add_argument("--max-episode-steps", type=int, default=150,
                         help="Must match the run's config header -- env default is 300.")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    # No hardcoded start/goal/jitter here on purpose: those live in
    # configs/<env_name>.yaml, so pointing at the same env_name used for
    # training guarantees this eval uses exactly the config the checkpoint
    # was trained and originally greedy-eval'd against.
    env = AirSimEnv(
        env_name=args.env_name,
        max_episode_steps=args.max_episode_steps,
        randomize_positions=True,
    )

    agent = D3RQNAgent(
        state_dim=env.STATE_DIM,
        action_dim=env.action_space.n,
        buffer_capacity=1,  # irrelevant for eval, required by constructor
    )
    agent.q_network.load_state_dict(torch.load(args.checkpoint, map_location=agent.device))
    agent.q_network.eval()

    successes, rewards, steps_list = [], [], []
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        hidden = agent.init_hidden()          # reset hidden state ONCE per episode
        done = False
        info = {}
        ep_reward = 0.0

        while not done:
            action, hidden = agent.select_action(state, hidden, greedy=True)  # hidden threaded forward
            state, reward, done, info = env.step(action)
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