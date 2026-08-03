import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """
    Plain MLP Q-network. For AirSim, input dim = your state feature
    vector length (position, velocity, orientation, goal distance,
    obstacle zone features) and output dim = 27 (3x3x3 discrete actions).
    For CartPole: input dim = 4, output dim = 2.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling architecture: shared trunk, then splits into a value stream
    V(s) (scalar) and an advantage stream A(s,a) (per-action), recombined as:
        Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))
    Subtracting the mean advantage keeps V and A identifiable (without it,
    the split is under-determined — infinitely many V/A combinations would
    give the same Q, which makes training unstable).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        features = self.trunk(x)
        value = self.value_stream(features)                      # (batch, 1)
        advantage = self.advantage_stream(features)               # (batch, action_dim)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """
    Vanilla DQN: single Q-network + target network, epsilon-greedy,
    uniform replay. Dueling/Double/PER/LSTM will be added as separate
    modifications on top of this once this baseline is validated.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
        target_update_freq=500,
        use_double=False,
        use_dueling=False,
        device=None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double = use_double    # toggle Double DQN target computation
        self.use_dueling = use_dueling  # toggle Dueling network architecture

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network_cls = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = network_cls(state_dim, action_dim).to(self.device)
        self.target_network = network_cls(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.loss_fn = nn.SmoothL1Loss()  # Huber loss, more stable than MSE

    def epsilon(self):
        # linear decay from epsilon_start to epsilon_end over epsilon_decay_steps
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, greedy=False):
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None  # not enough data yet

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device).unsqueeze(1)

        # current Q estimates for taken actions
        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            if self.use_double:
                # Double DQN: q_network SELECTS the best next action,
                # target_network EVALUATES it. Decoupling selection from
                # evaluation removes the systematic overestimation bias
                # that comes from using one network for both (the max
                # operator tends to pick actions whose Q-values are
                # inflated by noise, and evaluating with the same network
                # reinforces that inflation).
                best_next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, best_next_actions)
            else:
                # vanilla DQN target: max over target network's next-state Q-values
                next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]

            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()