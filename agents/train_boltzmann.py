# """
# Training script: D3QN + Boltzmann (softmax) exploration on AirSim Blocks.

# Separate experiment file — mirrors train_airsim.py but swaps in
# BoltzmannDQNAgent and logs temperature instead of epsilon, since that's
# what's actually driving exploration here.

# Usage:
#     python agents/train_boltzmann.py
# """

# import os
# import time
# import numpy as np
# import torch

# from airsim_env import AirSimEnv
# from boltzman_aget import BoltzmannDQNAgent  # <-- double check this matches your actual filename
# from logger import TrainingLogger

# # ---------------- config ----------------

# # GOAL_POSITION = (8816.915039, -435.996704, 1711.249756)      # adjust to a real reachable point in Blocks
# # START_POSITION = (3104.592773, 2863.085449, 1693.033203)

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
#         # Boltzmann exploration params (replace epsilon_* — they no longer
#         # do anything for action selection in BoltzmannDQNAgent).
#         temp_start=1.0,
#         temp_end=0.05,
#         temp_decay_steps=24000,   # matched to your old epsilon_decay_steps pacing
#         temp_schedule="linear",
#         target_update_freq=250,
#         use_double=True,
#         use_dueling=True,
#     )

#     agent = BoltzmannDQNAgent(**agent_config)

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
#         temperature = agent.current_temperature()

#         logger.log_episode(
#             episode=episode,
#             reward=round(episode_reward, 4),
#             avg10=round(recent_avg, 4),
#             success_rate10=round(success_rate, 2),
#             steps=info.get("step_num", 0),
#             epsilon="",  # unused for Boltzmann exploration — required arg on TrainingLogger, left blank
#             temperature=round(temperature, 4),   # goes through **extra as its own CSV column
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
#                 f"Temp: {temperature:.3f} | "          # <-- was Epsilon: {agent.epsilon():.3f}
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


# """
# Training script: D3QN + Boltzmann (softmax) exploration on AirSim Blocks.

# Separate experiment file — mirrors train_airsim.py but swaps in
# BoltzmannDQNAgent and logs temperature instead of epsilon, since that's
# what's actually driving exploration here.

# Usage:
#     python agents/train_boltzmann.py
# """

# import os
# import time
# import numpy as np
# import torch

# from airsim_env import AirSimEnv
# from boltzman_aget import BoltzmannDQNAgent  # <-- double check this matches your actual filename
# from logger import TrainingLogger

# # ---------------- config ----------------

# GOAL_POSITION = (20, 0, -3)      # adjust to a real reachable point in Blocks
# START_POSITION = (0, 0, -3)
# MAX_EPISODE_STEPS = 200          # keep modest — AirSim steps are real-time, not instant
# DT = 0.5
# SPEED = 2.0

# NUM_EPISODES = 200               # budgeted for ~3 hrs at observed ~40s/episode
# SAVE_EVERY = 10                  # checkpoint frequency (episodes)
# PRINT_EVERY = 1                  # AirSim episodes are slow — print every episode, not every 10

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# EVAL_EVERY = 20        # run a greedy eval block every N training episodes
# EVAL_EPISODES = 5      # keep small — each one costs real AirSim wall-clock time


# def run_greedy_eval(agent, env, num_episodes, logger, training_episode):
#     """
#     Runs num_episodes fully-greedy (argmax) episodes to measure the
#     underlying Q-function's quality, decoupled from whatever temperature
#     the training policy currently has. Does NOT write to the replay
#     buffer or call train_step() — this is measurement only, not training.
#     """
#     successes = []
#     for _ in range(num_episodes):
#         state = env.reset()
#         done = False
#         info = {}
#         while not done:
#             action_idx = agent.select_action_greedy(state)
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
#         # Boltzmann exploration params (replace epsilon_* — they no longer
#         # do anything for action selection in BoltzmannDQNAgent).
#         temp_start=1.0,
#         temp_end=0.05,
#         temp_decay_steps=24000,   # matched to your old epsilon_decay_steps pacing
#         temp_schedule="linear",
#         target_update_freq=250,
#         use_double=True,
#         use_dueling=True,
#     )

#     agent = BoltzmannDQNAgent(**agent_config)

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
#         temperature = agent.current_temperature()

#         logger.log_episode(
#             episode=episode,
#             reward=round(episode_reward, 4),
#             avg10=round(recent_avg, 4),
#             success_rate10=round(success_rate, 2),
#             steps=info.get("step_num", 0),
#             epsilon="",  # unused for Boltzmann exploration — required arg on TrainingLogger, left blank
#             temperature=round(temperature, 4),   # goes through **extra as its own CSV column
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
#                 f"Temp: {temperature:.3f} | "          # <-- was Epsilon: {agent.epsilon():.3f}
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

"""
Training script: D3QN + Boltzmann (softmax) exploration on AirSim Blocks.

Separate experiment file — mirrors train_airsim.py but swaps in
BoltzmannDQNAgent and logs temperature instead of epsilon, since that's
what's actually driving exploration here.

Usage:
    python agents/train_boltzmann.py
"""

import os
import time
import numpy as np
import torch

from airsim_env import AirSimEnv
from boltzman_aget import BoltzmannDQNAgent  # <-- double check this matches your actual filename
from logger import TrainingLogger

# ---------------- config ----------------

GOAL_POSITION = (20, 0, -3)      # adjust to a real reachable point in Blocks
START_POSITION = (0, 0, -3)
MAX_EPISODE_STEPS = 150          # keep modest — AirSim steps are real-time, not instant
DT = 0.5
SPEED = 2.0

NUM_EPISODES = 200               # budgeted for ~3 hrs at observed ~40s/episode
SAVE_EVERY = 10                  # checkpoint frequency (episodes)
PRINT_EVERY = 1                  # AirSim episodes are slow — print every episode, not every 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_EVERY = 20        # run a greedy eval block every N training episodes
EVAL_EPISODES = 5      # keep small — each one costs real AirSim wall-clock time


def run_greedy_eval(agent, env, num_episodes, logger, training_episode):
    """
    Runs num_episodes fully-greedy (argmax) episodes to measure the
    underlying Q-function's quality, decoupled from whatever temperature
    the training policy currently has. Does NOT write to the replay
    buffer or call train_step() — this is measurement only, not training.
    """
    successes = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        info = {}
        while not done:
            action_idx = agent.select_action_greedy(state)
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
    logger = TrainingLogger()  # prompts you to name this run
    RUN_ID = os.path.splitext(os.path.basename(logger.log_path))[0]

    env = AirSimEnv(
        goal_position=GOAL_POSITION,
        start_position=START_POSITION,
        max_episode_steps=MAX_EPISODE_STEPS,
        dt=DT,
        speed=SPEED,
    )

    agent_config = dict(
        state_dim=AirSimEnv.STATE_DIM,
        action_dim=env.action_space.n,
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=1_000_000,
        batch_size=32,
        # Boltzmann exploration params (replace epsilon_* — they no longer
        # do anything for action selection in BoltzmannDQNAgent).
        temp_start=1.0,
        temp_end=0.05,
        temp_decay_steps=24000,   # matched to your old epsilon_decay_steps pacing
        temp_schedule="linear",
        target_update_freq=250,
        use_double=True,
        use_dueling=True,
    )

    agent = BoltzmannDQNAgent(**agent_config)

    logger.log_config({
        "goal_position": GOAL_POSITION,
        "start_position": START_POSITION,
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
        temperature = agent.current_temperature()

        logger.log_episode(
            episode=episode,
            reward=round(episode_reward, 4),
            avg10=round(recent_avg, 4),
            success_rate10=round(success_rate, 2),
            steps=info.get("step_num", 0),
            epsilon="",  # unused for Boltzmann exploration — required arg on TrainingLogger, left blank
            temperature=round(temperature, 4),   # goes through **extra as its own CSV column
            start_position=episode_start_position,
            goal_position=episode_goal_position,
            elapsed_min=round(elapsed / 60, 2),
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
                f"Temp: {temperature:.3f} | "          # <-- was Epsilon: {agent.epsilon():.3f}
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