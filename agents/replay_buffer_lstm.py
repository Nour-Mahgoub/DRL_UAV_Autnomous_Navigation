# import random
# import numpy as np
# from collections import deque


# class SequenceReplayBuffer:
#     """
#     Episode-aware replay buffer for D3RQN (LSTM).

#     Unlike the flat per-transition D3QN buffer, this stores each episode as
#     its own list of transitions, and samples fixed-length contiguous windows
#     from within a single episode. This preserves temporal order, which the
#     LSTM needs to build a meaningful hidden state.

#     Usage during rollout:
#         buffer.start_episode()
#         for each step:
#             buffer.add_step(state, action, reward, next_state, done)
#         buffer.end_episode()   # commits the episode to the buffer

#     Usage during training:
#         states, actions, rewards, next_states, dones, mask = buffer.sample(batch_size, seq_len)
#         # each array has shape (batch_size, seq_len, ...)
#         # mask has shape (batch_size, seq_len) -- 1.0 for real steps, 0.0 for padding
#         # zero-init (h, c) at the start of each sampled sequence, unroll the LSTM
#         # across seq_len, and multiply the loss by `mask` so padded steps don't
#         # contribute to the gradient.
#     """

#     def __init__(self, capacity=2000):
#         # capacity here is in EPISODES, not transitions
#         self.episodes = deque(maxlen=capacity)
#         self._current_episode = []

#     def start_episode(self):
#         self._current_episode = []

#     def add_step(self, state, action, reward, next_state, done):
#         self._current_episode.append((state, action, reward, next_state, done))

#     def end_episode(self):
#         if len(self._current_episode) > 0:
#             self.episodes.append(self._current_episode)
#         self._current_episode = []

#     def __len__(self):
#         # number of stored episodes
#         return len(self.episodes)

#     def total_steps(self):
#         return sum(len(ep) for ep in self.episodes)

#     def sample(self, batch_size, seq_len):
#         """
#         Sample `batch_size` sequences of length `seq_len`, each drawn from a
#         single episode (never crossing episode boundaries).

#         Episodes shorter than seq_len are zero-padded at the end and flagged
#         via the mask. Episodes longer than seq_len have a random contiguous
#         window selected.
#         """
#         if len(self.episodes) == 0:
#             raise ValueError("Cannot sample from an empty buffer")

#         eligible = [ep for ep in self.episodes if len(ep) > 0]
#         episode_batch = random.choices(eligible, k=batch_size)

#         state_dim = np.asarray(episode_batch[0][0][0]).shape[0]

#         states = np.zeros((batch_size, seq_len, state_dim), dtype=np.float32)
#         next_states = np.zeros((batch_size, seq_len, state_dim), dtype=np.float32)
#         actions = np.zeros((batch_size, seq_len), dtype=np.int64)
#         rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
#         dones = np.zeros((batch_size, seq_len), dtype=np.float32)
#         mask = np.zeros((batch_size, seq_len), dtype=np.float32)

#         for i, ep in enumerate(episode_batch):
#             ep_len = len(ep)

#             if ep_len >= seq_len:
#                 start = random.randint(0, ep_len - seq_len)
#                 window = ep[start:start + seq_len]
#                 valid_len = seq_len
#             else:
#                 window = ep
#                 valid_len = ep_len

#             for t, (s, a, r, s2, d) in enumerate(window):
#                 states[i, t] = s
#                 actions[i, t] = a
#                 rewards[i, t] = r
#                 next_states[i, t] = s2
#                 dones[i, t] = float(d)
#                 mask[i, t] = 1.0
#             # steps beyond valid_len stay zero-padded, mask stays 0

#         return states, actions, rewards, next_states, dones, mask

import random
import numpy as np
from collections import deque


class SequenceReplayBuffer:
    """
    Episode-aware replay buffer for D3RQN (LSTM).

    Stores each episode as its own list of transitions and samples
    fixed-length contiguous windows from within a single episode, which
    preserves temporal order for the LSTM.

    Usage during rollout:
        buffer.start_episode()
        for each step:
            buffer.add_step(state, action, reward, next_state, done)
        buffer.end_episode()   # commits the episode to the buffer

    Usage during training:
        states, actions, rewards, next_states, dones, mask, burn_in_states, burn_in_mask = \
            buffer.sample(batch_size, seq_len, burn_in)

        # states/actions/rewards/next_states/dones/mask: shape (batch_size, seq_len, ...)
        #   mask is 1.0 for real steps, 0.0 for end-of-episode padding.
        # burn_in_states: shape (batch_size, burn_in, state_dim) -- the `burn_in`
        #   real states immediately preceding each sampled window, used ONLY to
        #   warm up (h, c) via a no-grad forward pass before training on the
        #   window itself. Left-zero-padded when a window starts too close to
        #   the beginning of its episode to have `burn_in` steps of history.
        # burn_in_mask: shape (batch_size, burn_in) -- 1.0 where burn_in_states
        #   holds a real prior transition, 0.0 where it's left-padding.
        #   (Not required to run the burn-in forward pass -- zero-padding a
        #   short prefix approximates "episode started a few steps early with
        #   no motion," which is a reasonable stand-in since burn-in never
        #   receives gradients -- but kept available for inspection/debugging.)
    """

    def __init__(self, capacity=2000):
        # capacity here is in EPISODES, not transitions
        self.episodes = deque(maxlen=capacity)
        self._current_episode = []

    def start_episode(self):
        self._current_episode = []

    def add_step(self, state, action, reward, next_state, done):
        self._current_episode.append((state, action, reward, next_state, done))

    def end_episode(self):
        if len(self._current_episode) > 0:
            self.episodes.append(self._current_episode)
        self._current_episode = []

    def __len__(self):
        # number of stored episodes
        return len(self.episodes)

    def total_steps(self):
        return sum(len(ep) for ep in self.episodes)

    def sample(self, batch_size, seq_len, burn_in=0):
        """
        Sample `batch_size` sequences of length `seq_len`, each drawn from a
        single episode (never crossing episode boundaries), plus up to
        `burn_in` real steps of history immediately preceding each window.

        Episodes shorter than seq_len are zero-padded at the end and flagged
        via `mask`. Episodes longer than seq_len have a random contiguous
        window selected. Windows that start fewer than `burn_in` steps into
        their episode get left-zero-padded burn-in (flagged via `burn_in_mask`).
        """
        if len(self.episodes) == 0:
            raise ValueError("Cannot sample from an empty buffer")

        eligible = [ep for ep in self.episodes if len(ep) > 0]
        episode_batch = random.choices(eligible, k=batch_size)

        state_dim = np.asarray(episode_batch[0][0][0]).shape[0]

        states = np.zeros((batch_size, seq_len, state_dim), dtype=np.float32)
        next_states = np.zeros((batch_size, seq_len, state_dim), dtype=np.float32)
        actions = np.zeros((batch_size, seq_len), dtype=np.int64)
        rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones = np.zeros((batch_size, seq_len), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)

        burn_in_states = np.zeros((batch_size, burn_in, state_dim), dtype=np.float32)
        burn_in_mask = np.zeros((batch_size, burn_in), dtype=np.float32)

        for i, ep in enumerate(episode_batch):
            ep_len = len(ep)

            if ep_len >= seq_len:
                start = random.randint(0, ep_len - seq_len)
                window = ep[start:start + seq_len]
                valid_len = seq_len
            else:
                start = 0
                window = ep
                valid_len = ep_len

            for t, (s, a, r, s2, d) in enumerate(window):
                states[i, t] = s
                actions[i, t] = a
                rewards[i, t] = r
                next_states[i, t] = s2
                dones[i, t] = float(d)
                mask[i, t] = 1.0
            # steps beyond valid_len stay zero-padded, mask stays 0

            if burn_in > 0 and start > 0:
                avail = min(burn_in, start)
                burn_window = ep[start - avail:start]
                offset = burn_in - avail  # left-pad with zeros if not enough history
                for t, (s, a, r, s2, d) in enumerate(burn_window):
                    burn_in_states[i, offset + t] = s
                    burn_in_mask[i, offset + t] = 1.0
            # if start == 0, burn_in_states stays all-zero/burn_in_mask all-0 --
            # correct, since the real rollout also had a zero-init hidden state here

        return states, actions, rewards, next_states, dones, mask, burn_in_states, burn_in_mask