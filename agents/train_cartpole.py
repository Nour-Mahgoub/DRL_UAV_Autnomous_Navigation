"""
CartPole validation run.

Goal: confirm the DQN implementation is correct BEFORE pointing it at AirSim.
Success criterion: average reward over last 20 episodes reaches ~195-200
(CartPole-v1 is considered "solved" around 195 avg over 100 episodes;
for a quick sanity check, a clear upward trend approaching 200 is enough
signal that the loop is correct).
"""

import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent

NUM_EPISODES = 400
PRINT_EVERY = 10


def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=20000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=8000,
        target_update_freq=250,
        use_double=True,
        use_dueling=True,
    )

    episode_rewards = []

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)

        if episode % PRINT_EVERY == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:6.1f} | "
                f"Avg(last 20): {avg_reward:6.1f} | "
                f"Epsilon: {agent.epsilon():.3f}"
            )

    env.close()

    final_avg = np.mean(episode_rewards[-20:])
    print(f"\nFinal avg reward (last 20 episodes): {final_avg:.1f}")

if __name__ == "__main__":
    main()