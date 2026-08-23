"""
Warm-start continuation run: loads q_network weights from a known-good
checkpoint (the d3qn_random_start_goal_nodepth_200 run, which reached 80%
success) instead of training from a randomly-initialized network. Combines
that head start with the new terminal success bonus in reward.py, which the
from-scratch rerun never got to test because it never once succeeded.

Replay buffer is intentionally started EMPTY, not restored — old transitions
were collected under the OLD reward function (no terminal bonus), so
replaying them would mix two different reward scales in one buffer.
Only the learned weights carry over, not past experience.

Usage:
    python agents/train_airsim_warmstart.py
"""

import os
import time
import numpy as np
import torch

from airsim_env import AirSimEnv
from dqn_agent import DQNAgent
from logger import TrainingLogger

# ---------------- config ----------------

GOAL_POSITION = (20, 0, -3)
START_POSITION = (0, 0, -3)
MAX_EPISODE_STEPS = 150
DT = 0.5
SPEED = 2.0

NUM_EPISODES = 150
SAVE_EVERY = 10
PRINT_EVERY = 1
EVAL_EVERY = 20
EVAL_EPISODES = 5

# Path to the checkpoint to warm-start from. Update this to the exact
# filename of your best-performing checkpoint from the successful
# d3qn_random_start_goal_nodepth_200 run.
WARM_START_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "d3qn_random_start_goal_nodepth_200_20260810_221150_ep200.pt"  # <-- verify this matches your actual saved filename
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


@torch.no_grad()
def select_action_greedy(agent, state):
    """Pure argmax over Q-values — no epsilon, no sampling. Eval only."""
    state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    q_values = agent.q_network(state_t)
    return int(torch.argmax(q_values, dim=-1).item())


def run_greedy_eval(agent, env, num_episodes, logger, training_episode):
    successes = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        info = {}
        while not done:
            action_idx = select_action_greedy(agent, state)
            state, _, done, info = env.step(action_idx)
        successes.append(info.get("is_success", False))

    eval_success_rate = 100.0 * sum(successes) / len(successes)
    logger.log_event(
        f"[greedy-eval @ episode {training_episode}] "
        f"success_rate={eval_success_rate:.1f}% over {num_episodes} episodes"
    )
    print(
        f"  >> Greedy eval @ ep {training_episode}: "
        f"{eval_success_rate:.1f}% success ({num_episodes} episodes)"
    )
    return eval_success_rate


def main():
    logger = TrainingLogger(results_dir=LOGS_DIR)
    RUN_ID = os.path.splitext(os.path.basename(logger.log_path))[0]

    env = AirSimEnv(
        goal_position=GOAL_POSITION,
        start_position=START_POSITION,
        max_episode_steps=MAX_EPISODE_STEPS,
        dt=DT,
        speed=SPEED,
        randomize_positions=True,
        start_jitter_radius=1.0,
        goal_jitter_radius=3.0,
    )

    agent_config = dict(
        state_dim=AirSimEnv.STATE_DIM,
        action_dim=env.action_space.n,
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=1_000_000,
        batch_size=32,
        epsilon_start=0.35,          # reduced — warm-started, not starting cold
        epsilon_end=0.2,
        epsilon_decay_steps=10000,   # shorter — less ground to re-explore
        target_update_freq=250,
        use_double=True,
        use_dueling=True,
    )

    agent = DQNAgent(**agent_config)

    # ---- warm start: load weights from the known-good checkpoint ----
    if not os.path.exists(WARM_START_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {WARM_START_CHECKPOINT}\n"
            "Update WARM_START_CHECKPOINT to your actual saved .pt filename."
        )
    state_dict = torch.load(WARM_START_CHECKPOINT, map_location=agent.device, weights_only=True)
    agent.q_network.load_state_dict(state_dict)
    # also sync the target network if the agent exposes one separately, so
    # bootstrapped targets start consistent with the warm-started online
    # network rather than against a freshly-initialized target net
    if hasattr(agent, "target_network"):
        agent.target_network.load_state_dict(state_dict)
        print("Synced target_network to match warm-started q_network.")
    print(f"Warm-started q_network from: {WARM_START_CHECKPOINT}")

    logger.log_config({
        "warm_started_from": WARM_START_CHECKPOINT,
        "goal_position": GOAL_POSITION,
        "start_position": START_POSITION,
        "randomize_positions": True,
        "start_jitter_radius": 1.0,
        "goal_jitter_radius": 3.0,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "dt": DT,
        "speed": SPEED,
        "state_dim": AirSimEnv.STATE_DIM,
        "action_dim": env.action_space.n,
        **agent_config,
    })

    episode_rewards = []
    episode_successes = []
    episode_step_counts = []
    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        episode_start_position = tuple(round(float(v), 3) for v in env.start_position)
        episode_goal_position = tuple(round(float(v), 3) for v in env.goal_position)
        episode_reward = 0
        done = False
        info = {}

        while not done:
            action_idx = agent.select_action(state)
            next_state, reward, done, info = env.step(action_idx)

            agent.store(state, action_idx, reward, next_state, float(done))
            loss = agent.train_step()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_successes.append(info.get("is_success", False))
        episode_step_counts.append(info.get("step_num", 0))

        recent_avg = np.mean(episode_rewards[-10:])
        success_rate = np.mean(episode_successes[-10:]) * 100
        elapsed = time.time() - start_time

        logger.log_episode(
            episode=episode,
            reward=round(episode_reward, 4),
            avg10=round(recent_avg, 4),
            success_rate10=round(success_rate, 2),
            steps=info.get("step_num", 0),
            epsilon=round(agent.epsilon(), 4),
            elapsed_min=round(elapsed / 60, 2),
            start_position=episode_start_position,
            goal_position=episode_goal_position,
            is_success=info.get("is_success", False),
            is_crash=info.get("is_crash", False),
            is_out_of_bounds=info.get("is_out_of_bounds", False),
            is_timeout=info.get("is_timeout", False),
        )

        if episode % PRINT_EVERY == 0:
            print(
                f"Ep {episode:4d} | Reward: {episode_reward:7.2f} | "
                f"Avg(last 10): {recent_avg:7.2f} | "
                f"Success%(last10): {success_rate:5.1f} | "
                f"Steps: {info.get('step_num', 0):3d} | "
                f"Epsilon: {agent.epsilon():.3f} | "
                f"Elapsed: {elapsed/60:.1f} min"
            )

        if episode % EVAL_EVERY == 0:
            run_greedy_eval(agent, env, EVAL_EPISODES, logger, episode)

        if episode % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_ep{episode}.pt")
            torch.save(agent.q_network.state_dict(), checkpoint_path)

            rewards_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_rewards.npy")
            np.save(rewards_path, np.array(episode_rewards))

    final_summary = (
        f"Training complete. {NUM_EPISODES} episodes in {(time.time()-start_time)/60:.1f} min. "
        f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}. "
        f"Final success rate (last 10): {np.mean(episode_successes[-10:])*100:.1f}%."
    )
    print(f"\n{final_summary}")
    logger.log_event(final_summary)
    logger.close()

    final_rewards_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_rewards_final.npy")
    np.save(final_rewards_path, np.array(episode_rewards))
    np.save(os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_steps_final.npy"), np.array(episode_step_counts))
    print(f"Saved reward history to {final_rewards_path}")
    print(f"Average steps/episode this run: {np.mean(episode_step_counts):.1f}")


if __name__ == "__main__":
    main()