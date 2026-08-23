# """
# Main training script: vanilla DQN on AirSim Blocks environment.

# This is your first real training run — the goal here is NOT necessarily
# full convergence today. A clean, uncrashed run with a training curve
# (even partial/rough) is a legitimate progress-report deliverable, since
# it demonstrates the full pipeline (env + agent + reward) works end to end.

# Usage:
#     python agents/train_airsim.py
# """

# import os
# import time
# import numpy as np
# import torch

# from airsim_env import AirSimEnv
# from dqn_agent import DQNAgent

# # ---------------- config ----------------

# import datetime

# RUN_NAME = "d3qn_segmentation"  # change this per run to avoid overwriting previous results
# RUN_ID = f"{RUN_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# GOAL_POSITION = (20, 0, -3)      # adjust to a real reachable point in Blocks
# START_POSITION = (0, 0, -3)
# MAX_EPISODE_STEPS = 150          # keep modest — AirSim steps are real-time, not instant
# DT = 0.5
# SPEED = 2.0

# NUM_EPISODES = 15              # budgeted for ~3 hrs at observed ~40s/episode
# SAVE_EVERY = 10                  # checkpoint frequency (episodes)
# PRINT_EVERY = 1                  # AirSim episodes are slow — print every episode, not every 10

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)


# def main():
#     env = AirSimEnv(
#         goal_position=GOAL_POSITION,
#         start_position=START_POSITION,
#         max_episode_steps=MAX_EPISODE_STEPS,
#         dt=DT,
#         speed=SPEED,
#     )

#     agent = DQNAgent(
#         state_dim=AirSimEnv.STATE_DIM,   # 15
#         action_dim=env.action_space.n,    # 27
#         # lr=1e-3,
#         lr=1e-4,
#         gamma=0.99,
#         # buffer_capacity=20000,
#         buffer_capacity=1000000,  # 1 million
#         # batch_size=64,
#         batch_size=32,
#         epsilon_start=1.0,
#         # epsilon_end=0.05,
#         epsilon_end=0.2,
#         # epsilon_decay_steps=23000,  # ~65% of expected total steps (400 ep x ~90 steps/ep est.)
#         epsilon_decay_steps=24000,  # 200x70
#         target_update_freq=250,
#         use_double=True,
#         use_dueling=True,
#     )

#     episode_rewards = []
#     episode_successes = []
#     episode_step_counts = []
#     start_time = time.time()

#     for episode in range(1, NUM_EPISODES + 1):
#         state = env.reset()
#         episode_reward = 0
#         done = False
#         info = {}

#         while not done:
#             action_idx = agent.select_action(state)
#             next_state, reward, done, info = env.step(action_idx)

#             agent.store(state, action_idx, reward, next_state, float(done))
#             loss = agent.train_step()

#             state = next_state
#             episode_reward += reward

#         episode_rewards.append(episode_reward)
#         episode_successes.append(info.get("is_success", False))
#         episode_step_counts.append(info.get("step_num", 0))

#         if episode % PRINT_EVERY == 0:
#             elapsed = time.time() - start_time
#             recent_avg = np.mean(episode_rewards[-10:])
#             success_rate = np.mean(episode_successes[-10:]) * 100
#             print(
#                 f"Ep {episode:4d} | Reward: {episode_reward:7.2f} | "
#                 f"Avg(last 10): {recent_avg:7.2f} | "
#                 f"Success%(last10): {success_rate:5.1f} | "
#                 f"Steps: {info.get('step_num', 0):3d} | "
#                 f"Epsilon: {agent.epsilon():.3f} | "
#                 f"Elapsed: {elapsed/60:.1f} min"
#             )

#         if episode % SAVE_EVERY == 0:
#             checkpoint_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_ep{episode}.pt")
#             torch.save(agent.q_network.state_dict(), checkpoint_path)

#             rewards_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_rewards.npy")
#             np.save(rewards_path, np.array(episode_rewards))

#     print(f"\nTraining complete. {NUM_EPISODES} episodes in {(time.time()-start_time)/60:.1f} min.")
#     print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
#     print(f"Final success rate (last 10): {np.mean(episode_successes[-10:])*100:.1f}%")

#     final_rewards_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_rewards_final.npy")
#     np.save(final_rewards_path, np.array(episode_rewards))
#     np.save(os.path.join(RESULTS_DIR, f"{RUN_ID}_episode_steps_final.npy"), np.array(episode_step_counts))
#     print(f"Saved reward history to {final_rewards_path}")
#     print(f"Average steps/episode this run: {np.mean(episode_step_counts):.1f}")


# if __name__ == "__main__":
#     main()

"""
Main training script: vanilla DQN on AirSim Blocks environment.

This is your first real training run — the goal here is NOT necessarily
full convergence today. A clean, uncrashed run with a training curve
(even partial/rough) is a legitimate progress-report deliverable, since
it demonstrates the full pipeline (env + agent + reward) works end to end.

Usage:
    python agents/train_airsim.py
"""

# import os
# import time
# import numpy as np
# import torch

# from airsim_env import AirSimEnv
# from dqn_agent import DQNAgent

# # ---------------- config ----------------

# from logger import TrainingLogger

# GOAL_POSITION = (20, 0, -3)      # adjust to a real reachable point in Blocks
# START_POSITION = (0, 0, -3)
# MAX_EPISODE_STEPS = 150          # keep modest — AirSim steps are real-time, not instant
# DT = 0.5
# SPEED = 2.0

# NUM_EPISODES = 200               # budgeted for ~3 hrs at observed ~40s/episode
# SAVE_EVERY = 10                  # checkpoint frequency (episodes)
# PRINT_EVERY = 1                  # AirSim episodes are slow — print every episode, not every 10

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)


# def main():
#     logger = TrainingLogger()  # prompts you to name this run
#     RUN_ID = os.path.splitext(os.path.basename(logger.log_path))[0]

#     env = AirSimEnv(
#         goal_position=GOAL_POSITION,
#         start_position=START_POSITION,
#         max_episode_steps=MAX_EPISODE_STEPS,
#         dt=DT,
#         speed=SPEED,
#     )

#     agent_config = dict(
#         state_dim=AirSimEnv.STATE_DIM,
#         action_dim=env.action_space.n,
#         lr=1e-4,
#         gamma=0.99,
#         buffer_capacity=1_000_000,
#         batch_size=32,
#         epsilon_start=1.0,
#         epsilon_end=0.2,
#         epsilon_decay_steps=24000,
#         target_update_freq=250,
#         use_double=True,
#         use_dueling=True,
#     )

#     agent = DQNAgent(**agent_config)

#     logger.log_config({
#         "goal_position": GOAL_POSITION,
#         "start_position": START_POSITION,
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
#         episode_reward = 0
#         done = False
#         info = {}

#         while not done:
#             action_idx = agent.select_action(state)
#             next_state, reward, done, info = env.step(action_idx)

#             agent.store(state, action_idx, reward, next_state, float(done))
#             loss = agent.train_step()

#             state = next_state
#             episode_reward += reward

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

###########Randm start and goal log per episode 
import os
import time
import numpy as np
import torch

from airsim_env import AirSimEnv
from dqn_agent import DQNAgent

# ---------------- config ----------------

from logger import TrainingLogger

GOAL_POSITION = (20, 0, -3)      # adjust to a real reachable point in Blocks
START_POSITION = (0, 0, -3)
MAX_EPISODE_STEPS = 150          # keep modest — AirSim steps are real-time, not instant
DT = 0.5
SPEED = 2.0

NUM_EPISODES = 300               # budgeted for ~3 hrs at observed ~40s/episode
SAVE_EVERY = 10                  # checkpoint frequency (episodes)
PRINT_EVERY = 1                  # AirSim episodes are slow — print every episode, not every 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

EVAL_EVERY = 20        # run a greedy eval block every N training episodes
EVAL_EPISODES = 5      # keep small — each one costs real AirSim wall-clock time


@torch.no_grad()
def select_action_greedy(agent, state):
    """Pure argmax over Q-values — no epsilon, no sampling. Eval only."""
    state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    q_values = agent.q_network(state_t)
    return int(torch.argmax(q_values, dim=-1).item())


def run_greedy_eval(agent, env, num_episodes, logger, training_episode):
    """
    Runs num_episodes fully-greedy episodes to measure the underlying
    Q-function's quality, decoupled from wherever epsilon currently sits.
    Does NOT write to the replay buffer or call train_step().
    """
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
    logger = TrainingLogger(results_dir=LOGS_DIR)  # prompts you to name this run
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
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=24000,  
        target_update_freq=250,
        use_double=True,
        use_dueling=True,
    )

    agent = DQNAgent(**agent_config)

    logger.log_config({
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
        # capture the ACTUAL position used this episode — if randomize_positions
        # or scenario_pairs is set on the env, this differs from GOAL_POSITION/
        # START_POSITION above, which are only the base/reference values.
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