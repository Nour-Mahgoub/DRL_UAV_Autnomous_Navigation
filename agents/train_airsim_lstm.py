# """
# Training script: D3RQN (Dueling + Double + Recurrent DQN) on AirSim Blocks.

# Adapted from train_airsim.py. Key differences from the D3QN version:
#   - hidden state (h, c) is created per episode and threaded through every
#     select_action() call
#   - agent.start_episode() / agent.end_episode() bracket each episode so the
#     SequenceReplayBuffer commits it as one contiguous unit
#   - greedy eval routes through agent.select_action(..., greedy=True) instead
#     of calling agent.q_network(...) directly, since the network now expects
#     a (batch, seq_len, state_dim) input plus a hidden state

# Usage:
#     python agents/train_airsim_lstm.py
# """

# import os
# import time
# import numpy as np
# import torch

# from airsim_env import AirSimEnv
# from d3rqn_agent import D3RQNAgent
# from logger import TrainingLogger

# # ---------------- config ----------------

# GOAL_POSITION = (30.0, 0.0, -3.0)
# START_POSITION = (0, 0, -3)
# MAX_EPISODE_STEPS = 150
# DT = 0.5
# SPEED = 2.0

# NUM_EPISODES = 300
# SAVE_EVERY = 10
# PRINT_EVERY = 1

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
# LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
# os.makedirs(RESULTS_DIR, exist_ok=True)
# os.makedirs(LOGS_DIR, exist_ok=True)

# EVAL_EVERY = 20
# EVAL_EPISODES = 5


# def run_greedy_eval(agent, env, num_episodes, logger, training_episode):
#     """
#     Runs num_episodes fully-greedy episodes to measure the underlying
#     Q-function's quality, decoupled from wherever epsilon currently sits.
#     Does NOT write to the replay buffer or call train_step().
#     Each eval episode gets its own fresh hidden state.
#     """
#     successes = []
#     for _ in range(num_episodes):
#         state = env.reset()
#         hidden = agent.init_hidden()
#         done = False
#         info = {}
#         while not done:
#             action_idx, hidden = agent.select_action(state, hidden, greedy=True)
#             state, _, done, info = env.step(action_idx)
#         successes.append(info.get("is_success", False))

#     eval_success_rate = 100.0 * sum(successes) / len(successes)
#     logger.log_event(
#         f"[greedy-eval @ episode {training_episode}] "
#         f"success_rate={eval_success_rate:.1f}% over {num_episodes} episodes"
#     )
#     print(
#         f"  >> Greedy eval @ ep {training_episode}: "
#         f"{eval_success_rate:.1f}% success ({num_episodes} episodes)"
#     )
#     return eval_success_rate


# def main():
#     logger = TrainingLogger(results_dir=LOGS_DIR)
#     RUN_ID = os.path.splitext(os.path.basename(logger.log_path))[0]

#     env = AirSimEnv(
#         goal_position=GOAL_POSITION,
#         start_position=START_POSITION,
#         max_episode_steps=MAX_EPISODE_STEPS,
#         dt=DT,
#         speed=SPEED,
#         randomize_positions=True,
#         start_jitter_radius=2.0,
#         goal_jitter_radius=5.0,
#     )

#     agent_config = dict(
#         state_dim=AirSimEnv.STATE_DIM,
#         action_dim=env.action_space.n,
#         lr=1e-4,
#         gamma=0.99,
#         buffer_capacity=5000,     # episodes, not transitions -- see d3rqn_agent.py notes
#         batch_size=32,
#         seq_len=20,
#         epsilon_start=1.0,
#         epsilon_end=0.2,
#         epsilon_decay_steps=24000,
#         target_update_freq=250,
#         use_double=True,
#     )

#     agent = D3RQNAgent(**agent_config)

#     logger.log_config({
#         "architecture": "D3RQN",
#         "environment": env.env_name,
#         "goal_position": GOAL_POSITION,
#         "start_position": START_POSITION,
#         "randomize_positions": True,
#         "start_jitter_radius": env.start_jitter_radius,
#         "goal_jitter_radius": env.goal_jitter_radius,
#         "max_episode_steps": MAX_EPISODE_STEPS,
#         "dt": DT,
#         "speed": SPEED,
#         "state_dim": AirSimEnv.STATE_DIM,
#         "action_dim": env.action_space.n,
#         **agent_config,
#     })

#     episode_rewards = []
#     episode_successes = []
#     episode_step_counts = []
#     start_time = time.time()

#     for episode in range(1, NUM_EPISODES + 1):
#         state = env.reset()
#         episode_start_position = tuple(round(float(v), 3) for v in env.start_position)
#         episode_goal_position = tuple(round(float(v), 3) for v in env.goal_position)

#         hidden = agent.init_hidden()
#         agent.start_episode()

#         episode_reward = 0
#         done = False
#         info = {}

#         while not done:
#             action_idx, hidden = agent.select_action(state, hidden)
#             next_state, reward, done, info = env.step(action_idx)

#             agent.store(state, action_idx, reward, next_state, float(done))
#             loss = agent.train_step()  # trains on already-completed episodes in the buffer

#             state = next_state
#             episode_reward += reward

#         agent.end_episode()  # commits this episode to the buffer

#         episode_rewards.append(episode_reward)
#         episode_successes.append(info.get("is_success", False))
#         episode_step_counts.append(info.get("step_num", 0))

#         recent_avg = np.mean(episode_rewards[-10:])
#         success_rate = np.mean(episode_successes[-10:]) * 100
#         elapsed = time.time() - start_time

#         logger.log_episode(
#             episode=episode,
#             reward=round(episode_reward, 4),
#             avg10=round(recent_avg, 4),
#             success_rate10=round(success_rate, 2),
#             steps=info.get("step_num", 0),
#             epsilon=round(agent.epsilon(), 4),
#             elapsed_min=round(elapsed / 60, 2),
#             start_position=episode_start_position,
#             goal_position=episode_goal_position,
#             is_success=info.get("is_success", False),
#             is_crash=info.get("is_crash", False),
#             is_out_of_bounds=info.get("is_out_of_bounds", False),
#             is_timeout=info.get("is_timeout", False),
#         )

#         if episode % PRINT_EVERY == 0:
#             print(
#                 f"Ep {episode:4d} | Reward: {episode_reward:7.2f} | "
#                 f"Avg(last 10): {recent_avg:7.2f} | "
#                 f"Success%(last10): {success_rate:5.1f} | "
#                 f"Steps: {info.get('step_num', 0):3d} | "
#                 f"Epsilon: {agent.epsilon():.3f} | "
#                 f"Elapsed: {elapsed/60:.1f} min"
#             )

#         if episode % EVAL_EVERY == 0:
#             run_greedy_eval(agent, env, EVAL_EPISODES, logger, episode)

#         if episode % SAVE_EVERY == 0:
#             checkpoint_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_ep{episode}.pt")
#             torch.save(agent.q_network.state_dict(), checkpoint_path)

#             rewards_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_rewards.npy")
#             np.save(rewards_path, np.array(episode_rewards))

#     final_summary = (
#         f"Training complete. {NUM_EPISODES} episodes in {(time.time()-start_time)/60:.1f} min. "
#         f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}. "
#         f"Final success rate (last 10): {np.mean(episode_successes[-10:])*100:.1f}%."
#     )
#     print(f"\n{final_summary}")
#     logger.log_event(final_summary)
#     logger.close()

#     final_rewards_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_rewards_final.npy")
#     np.save(final_rewards_path, np.array(episode_rewards))
#     np.save(os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_steps_final.npy"), np.array(episode_step_counts))
#     print(f"Saved reward history to {final_rewards_path}")
#     print(f"Average steps/episode this run: {np.mean(episode_step_counts):.1f}")


# if __name__ == "__main__":
#     main()

import os
import time
import numpy as np
import torch

from airsim_env import AirSimEnv
from d3rqn_agent import D3RQNAgent
from logger import TrainingLogger

# ---------------- config ----------------

GOAL_POSITION = (30.0, 0.0, -3.0)
START_POSITION = (0, 0, -3)
MAX_EPISODE_STEPS = 150
DT = 0.5
SPEED = 2.0

NUM_EPISODES = 300
SAVE_EVERY = 10
PRINT_EVERY = 1

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

EVAL_EVERY = 20
EVAL_EPISODES = 5


def run_greedy_eval(agent, env, num_episodes, logger, training_episode):
    """
    Runs num_episodes fully-greedy episodes to measure the underlying
    Q-function's quality, decoupled from wherever epsilon currently sits.
    Does NOT write to the replay buffer or call train_step().
    Each eval episode gets its own fresh hidden state.
    """
    successes = []
    for _ in range(num_episodes):
        state = env.reset()
        hidden = agent.init_hidden()
        done = False
        info = {}
        while not done:
            action_idx, hidden = agent.select_action(state, hidden, greedy=True)
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
        start_jitter_radius=2.0,
        goal_jitter_radius=5.0,
    )

    agent_config = dict(
        state_dim=AirSimEnv.STATE_DIM,
        action_dim=env.action_space.n,
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=5000,     # episodes, not transitions -- see d3rqn_agent.py notes
        batch_size=32,
        seq_len=20,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=24000,
        target_update_freq=250,
        use_double=True,
    )

    agent = D3RQNAgent(**agent_config)

    logger.log_config({
        "architecture": "D3RQN",
        "environment": env.env_name,
        "goal_position": GOAL_POSITION,
        "start_position": START_POSITION,
        "randomize_positions": True,
        "start_jitter_radius": env.start_jitter_radius,
        "goal_jitter_radius": env.goal_jitter_radius,
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

        hidden = agent.init_hidden()
        agent.start_episode()

        episode_reward = 0
        done = False
        info = {}

        # diagnostics accumulated over this episode's train_step() calls
        episode_losses = []
        episode_mean_abs_qs = []
        episode_grad_norms = []

        while not done:
            action_idx, hidden = agent.select_action(state, hidden)
            next_state, reward, done, info = env.step(action_idx)

            agent.store(state, action_idx, reward, next_state, float(done))
            train_info = agent.train_step()  # trains on already-completed episodes in the buffer
            if train_info is not None:
                episode_losses.append(train_info["loss"])
                episode_mean_abs_qs.append(train_info["mean_abs_q"])
                episode_grad_norms.append(train_info["grad_norm"])

            state = next_state
            episode_reward += reward

        agent.end_episode()  # commits this episode to the buffer

        episode_rewards.append(episode_reward)
        episode_successes.append(info.get("is_success", False))
        episode_step_counts.append(info.get("step_num", 0))

        recent_avg = np.mean(episode_rewards[-10:])
        success_rate = np.mean(episode_successes[-10:]) * 100
        elapsed = time.time() - start_time

        avg_loss = float(np.mean(episode_losses)) if episode_losses else None
        avg_mean_abs_q = float(np.mean(episode_mean_abs_qs)) if episode_mean_abs_qs else None
        max_grad_norm = float(np.max(episode_grad_norms)) if episode_grad_norms else None

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

        # separate event line (not a CSV column) with the diagnostics that
        # matter for catching a training collapse AS it happens rather than
        # inferring it after the fact from reward -- e.g. avg_mean_abs_q
        # growing without bound points at value blowup; max_grad_norm sitting
        # at the clip ceiling (10.0) repeatedly points at exploding gradients
        # the clip is masking rather than preventing.
        if avg_loss is not None:
            logger.log_event(
                f"[diag @ episode {episode}] "
                f"avg_loss={avg_loss:.4f} avg_mean_abs_q={avg_mean_abs_q:.4f} "
                f"max_grad_norm={max_grad_norm:.4f}"
            )

        if episode % PRINT_EVERY == 0:
            diag_str = (
                f" | Loss: {avg_loss:.3f} | MeanQ: {avg_mean_abs_q:.2f} | GradNorm: {max_grad_norm:.2f}"
                if avg_loss is not None else ""
            )
            print(
                f"Ep {episode:4d} | Reward: {episode_reward:7.2f} | "
                f"Avg(last 10): {recent_avg:7.2f} | "
                f"Success%(last10): {success_rate:5.1f} | "
                f"Steps: {info.get('step_num', 0):3d} | "
                f"Epsilon: {agent.epsilon():.3f} | "
                f"Elapsed: {elapsed/60:.1f} min"
                f"{diag_str}"
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