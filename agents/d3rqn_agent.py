# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from replay_buffer_lstm import SequenceReplayBuffer


# class RecurrentDuelingQNetwork(nn.Module):
#     """
#     D3RQN network: FC encoder -> LSTM -> dueling streams (V, A).

#     forward() operates on whole sequences:
#         x:      (batch, seq_len, state_dim)
#         hidden: (h, c), each (lstm_layers, batch, hidden_dim), or None to zero-init
#     returns:
#         q_values:   (batch, seq_len, action_dim)
#         new_hidden: (h, c) after processing the sequence
#     """

#     def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=1):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm_layers = lstm_layers

#         self.encoder = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
#         self.value_stream = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#         )
#         self.advantage_stream = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#         )

#     def init_hidden(self, batch_size, device):
#         h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
#         c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
#         return (h, c)

#     def forward(self, x, hidden=None):
#         batch_size = x.size(0)
#         if hidden is None:
#             hidden = self.init_hidden(batch_size, x.device)

#         features = self.encoder(x)                    # (batch, seq_len, hidden_dim)
#         lstm_out, new_hidden = self.lstm(features, hidden)  # (batch, seq_len, hidden_dim)

#         value = self.value_stream(lstm_out)            # (batch, seq_len, 1)
#         advantage = self.advantage_stream(lstm_out)     # (batch, seq_len, action_dim)
#         q_values = value + (advantage - advantage.mean(dim=2, keepdim=True))
#         return q_values, new_hidden


# class D3RQNAgent:
#     """
#     Dueling + Double + Recurrent DQN agent. PER is NOT included yet
#     (naive per-transition PER is incompatible with sequence sampling anyway
#     - see the R2D2 problem notes).

#     burn_in: number of real steps of history run through the network with
#     no_grad before each training window, to warm up (h, c) to a realistic
#     state instead of zero-initializing mid-episode windows. 0 disables it
#     and reproduces the old (buggy) always-zero-init behavior. See
#     Kapturowski et al. 2019 (R2D2) for the burn-in vs. stored-state
#     discussion -- this implements burn-in, the simpler of the two.
#     """

#     def __init__(
#         self,
#         state_dim,
#         action_dim,
#         lr=1e-3,
#         gamma=0.99,
#         buffer_capacity=2000,      # episodes, not transitions
#         batch_size=32,
#         seq_len=20,
#         burn_in=10,
#         epsilon_start=1.0,
#         epsilon_end=0.05,
#         epsilon_decay_steps=24000,
#         target_update_freq=500,
#         use_double=True,
#         hidden_dim=128,
#         lstm_layers=1,
#         device=None,
#     ):
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.seq_len = seq_len
#         self.burn_in = burn_in
#         self.target_update_freq = target_update_freq
#         self.use_double = use_double

#         self.epsilon_start = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay_steps = epsilon_decay_steps
#         self.steps_done = 0

#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.q_network = RecurrentDuelingQNetwork(
#             state_dim, action_dim, hidden_dim, lstm_layers
#         ).to(self.device)
#         self.target_network = RecurrentDuelingQNetwork(
#             state_dim, action_dim, hidden_dim, lstm_layers
#         ).to(self.device)
#         self.target_network.load_state_dict(self.q_network.state_dict())
#         self.target_network.eval()

#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
#         self.buffer = SequenceReplayBuffer(buffer_capacity)

#     def epsilon(self):
#         frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
#         return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

#     def init_hidden(self):
#         """Call at the start of each rollout episode; pass the result into select_action."""
#         return self.q_network.init_hidden(batch_size=1, device=self.device)

#     def select_action(self, state, hidden, greedy=False):
#         """
#         Returns (action, new_hidden). Hidden state always advances based on
#         the real observation, whether or not the returned action is
#         exploratory -- exploration should not create gaps in the LSTM's
#         temporal context.
#         """
#         eps = 0.0 if greedy else self.epsilon()

#         with torch.no_grad():
#             state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
#             q_values, new_hidden = self.q_network(state_t, hidden)

#         if random.random() < eps:
#             action = random.randrange(self.action_dim)
#         else:
#             action = int(q_values[0, 0].argmax().item())

#         return action, new_hidden

#     # --- episode-level buffer interface (delegates to SequenceReplayBuffer) ---
#     def start_episode(self):
#         self.buffer.start_episode()

#     def store(self, state, action, reward, next_state, done):
#         self.buffer.add_step(state, action, reward, next_state, done)

#     def end_episode(self):
#         self.buffer.end_episode()

#     def _burn_in_hidden(self, burn_in_states_np):
#         """
#         Runs the burn-in slice through both networks with no_grad to produce
#         realistic starting hidden states for the training window. Returns
#         (hidden_for_online, hidden_for_target), each None if burn_in == 0.
#         """
#         if self.burn_in == 0:
#             return None, None

#         burn_in_states = torch.tensor(burn_in_states_np, device=self.device)  # (B, burn_in, state_dim)
#         with torch.no_grad():
#             _, hidden = self.q_network(burn_in_states, None)
#             _, target_hidden = self.target_network(burn_in_states, None)
#         return hidden, target_hidden

#     def train_step(self):
#         if len(self.buffer) < self.batch_size:
#             return None  # not enough episodes yet

#         states, actions, rewards, next_states, dones, mask, burn_in_states, _burn_in_mask = \
#             self.buffer.sample(self.batch_size, self.seq_len, self.burn_in)

#         states = torch.tensor(states, device=self.device)              # (B, T, state_dim)
#         next_states = torch.tensor(next_states, device=self.device)    # (B, T, state_dim)
#         actions = torch.tensor(actions, device=self.device).unsqueeze(-1)   # (B, T, 1)
#         rewards = torch.tensor(rewards, device=self.device).unsqueeze(-1)   # (B, T, 1)
#         dones = torch.tensor(dones, device=self.device).unsqueeze(-1)       # (B, T, 1)
#         mask = torch.tensor(mask, device=self.device).unsqueeze(-1)         # (B, T, 1)

#         # warm up (h, c) from real preceding history instead of zero-init,
#         # unless the window starts at step 0 of its episode (then zero-init
#         # IS correct, and burn_in_states is already all-zero to match)
#         hidden, target_hidden = self._burn_in_hidden(burn_in_states)

#         q_values, _ = self.q_network(states, hidden)
#         q_values = q_values.gather(2, actions)  # (B, T, 1)

#         with torch.no_grad():
#             # NOTE: next_states is processed as its own parallel unroll from
#             # the same warmed-up hidden state as `states`, not chained after
#             # it step-by-step -- same simplification the original zero-init
#             # code used, just now starting from a realistic hidden state
#             # instead of always-zero. Fully chaining next_states after each
#             # real step would be more faithful but is a larger restructuring;
#             # flag if you want that version too.
#             if self.use_double:
#                 next_q_online, _ = self.q_network(next_states, hidden)
#                 best_next_actions = next_q_online.argmax(dim=2, keepdim=True)
#                 next_q_target, _ = self.target_network(next_states, target_hidden)
#                 next_q_values = next_q_target.gather(2, best_next_actions)
#             else:
#                 next_q_target, _ = self.target_network(next_states, target_hidden)
#                 next_q_values = next_q_target.max(dim=2, keepdim=True)[0]

#             targets = rewards + self.gamma * next_q_values * (1 - dones)

#         # per-step Huber loss, masked so padded steps don't contribute
#         loss_per_step = F.smooth_l1_loss(q_values, targets, reduction="none")  # (B, T, 1)
#         loss = (loss_per_step * mask).sum() / mask.sum().clamp(min=1.0)

#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
#         self.optimizer.step()

#         self.steps_done += 1
#         if self.steps_done % self.target_update_freq == 0:
#             self.target_network.load_state_dict(self.q_network.state_dict())

#         return loss.item()

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer_lstm import SequenceReplayBuffer


class RecurrentDuelingQNetwork(nn.Module):
    """
    D3RQN network: FC encoder -> LSTM -> dueling streams (V, A).

    forward() operates on whole sequences:
        x:      (batch, seq_len, state_dim)
        hidden: (h, c), each (lstm_layers, batch, hidden_dim), or None to zero-init
    returns:
        q_values:   (batch, seq_len, action_dim)
        new_hidden: (h, c) after processing the sequence
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
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

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        features = self.encoder(x)                    # (batch, seq_len, hidden_dim)
        lstm_out, new_hidden = self.lstm(features, hidden)  # (batch, seq_len, hidden_dim)

        value = self.value_stream(lstm_out)            # (batch, seq_len, 1)
        advantage = self.advantage_stream(lstm_out)     # (batch, seq_len, action_dim)
        q_values = value + (advantage - advantage.mean(dim=2, keepdim=True))
        return q_values, new_hidden


class D3RQNAgent:
    """
    Dueling + Double + Recurrent DQN agent. PER is NOT included yet
    (naive per-transition PER is incompatible with sequence sampling anyway
    - see the R2D2 problem notes).

    burn_in: number of real steps of history run through the network with
    no_grad before each training window, to warm up (h, c) to a realistic
    state instead of zero-initializing mid-episode windows. 0 disables it
    and reproduces the old (buggy) always-zero-init behavior. See
    Kapturowski et al. 2019 (R2D2) for the burn-in vs. stored-state
    discussion -- this implements burn-in, the simpler of the two.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=2000,      # episodes, not transitions
        batch_size=32,
        seq_len=20,
        burn_in=10,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=24000,
        target_update_freq=500,
        use_double=True,
        hidden_dim=128,
        lstm_layers=1,
        device=None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.target_update_freq = target_update_freq
        self.use_double = use_double

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = RecurrentDuelingQNetwork(
            state_dim, action_dim, hidden_dim, lstm_layers
        ).to(self.device)
        self.target_network = RecurrentDuelingQNetwork(
            state_dim, action_dim, hidden_dim, lstm_layers
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = SequenceReplayBuffer(buffer_capacity)

    def epsilon(self):
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def init_hidden(self):
        """Call at the start of each rollout episode; pass the result into select_action."""
        return self.q_network.init_hidden(batch_size=1, device=self.device)

    def select_action(self, state, hidden, greedy=False):
        """
        Returns (action, new_hidden). Hidden state always advances based on
        the real observation, whether or not the returned action is
        exploratory -- exploration should not create gaps in the LSTM's
        temporal context.
        """
        eps = 0.0 if greedy else self.epsilon()

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
            q_values, new_hidden = self.q_network(state_t, hidden)

        if random.random() < eps:
            action = random.randrange(self.action_dim)
        else:
            action = int(q_values[0, 0].argmax().item())

        return action, new_hidden

    # --- episode-level buffer interface (delegates to SequenceReplayBuffer) ---
    def start_episode(self):
        self.buffer.start_episode()

    def store(self, state, action, reward, next_state, done):
        self.buffer.add_step(state, action, reward, next_state, done)

    def end_episode(self):
        self.buffer.end_episode()

    def _burn_in_hidden(self, burn_in_states_np):
        """
        Runs the burn-in slice through both networks with no_grad to produce
        realistic starting hidden states for the training window. Returns
        (hidden_for_online, hidden_for_target), each None if burn_in == 0.
        """
        if self.burn_in == 0:
            return None, None

        burn_in_states = torch.tensor(burn_in_states_np, device=self.device)  # (B, burn_in, state_dim)
        with torch.no_grad():
            _, hidden = self.q_network(burn_in_states, None)
            _, target_hidden = self.target_network(burn_in_states, None)
        return hidden, target_hidden

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None  # not enough episodes yet

        states, actions, rewards, next_states, dones, mask, burn_in_states, _burn_in_mask = \
            self.buffer.sample(self.batch_size, self.seq_len, self.burn_in)

        states = torch.tensor(states, device=self.device)              # (B, T, state_dim)
        next_states = torch.tensor(next_states, device=self.device)    # (B, T, state_dim)
        actions = torch.tensor(actions, device=self.device).unsqueeze(-1)   # (B, T, 1)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(-1)   # (B, T, 1)
        dones = torch.tensor(dones, device=self.device).unsqueeze(-1)       # (B, T, 1)
        mask = torch.tensor(mask, device=self.device).unsqueeze(-1)         # (B, T, 1)

        # warm up (h, c) from real preceding history instead of zero-init,
        # unless the window starts at step 0 of its episode (then zero-init
        # IS correct, and burn_in_states is already all-zero to match)
        hidden, target_hidden = self._burn_in_hidden(burn_in_states)

        q_values, _ = self.q_network(states, hidden)
        q_values = q_values.gather(2, actions)  # (B, T, 1)

        with torch.no_grad():
            # NOTE: next_states is processed as its own parallel unroll from
            # the same warmed-up hidden state as `states`, not chained after
            # it step-by-step -- same simplification the original zero-init
            # code used, just now starting from a realistic hidden state
            # instead of always-zero. Fully chaining next_states after each
            # real step would be more faithful but is a larger restructuring;
            # flag if you want that version too.
            if self.use_double:
                next_q_online, _ = self.q_network(next_states, hidden)
                best_next_actions = next_q_online.argmax(dim=2, keepdim=True)
                next_q_target, _ = self.target_network(next_states, target_hidden)
                next_q_values = next_q_target.gather(2, best_next_actions)
            else:
                next_q_target, _ = self.target_network(next_states, target_hidden)
                next_q_values = next_q_target.max(dim=2, keepdim=True)[0]

            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # per-step Huber loss, masked so padded steps don't contribute
        loss_per_step = F.smooth_l1_loss(q_values, targets, reduction="none")  # (B, T, 1)
        loss = (loss_per_step * mask).sum() / mask.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # diagnostics: mean_abs_q and grad_norm let you catch value blowup or
        # exploding gradients AS training happens, instead of only inferring
        # it after the fact from a reward collapse several episodes later.
        # mean_abs_q is computed only over real (masked) steps.
        with torch.no_grad():
            mean_abs_q = (q_values.abs() * mask).sum() / mask.sum().clamp(min=1.0)

        return {
            "loss": loss.item(),
            "mean_abs_q": mean_abs_q.item(),
            "grad_norm": float(grad_norm),
        }