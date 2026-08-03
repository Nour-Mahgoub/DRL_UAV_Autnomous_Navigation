# """
# boltzmann_agent.py

# Standalone Boltzmann (softmax) exploration experiment.

# Does NOT modify dqn_agent.py. This subclasses DQNAgent and overrides
# ONLY the action-selection method, so the network architecture, replay
# buffer, and training/update logic stay exactly as they are in your
# working D3QN implementation.

# --------------------------------------------------------------------
# ASSUMPTIONS ABOUT YOUR DQNAgent (from dqn_agent.py) -- fix if wrong:
#   - class DQNAgent(...) lives in dqn_agent.py
#   - self.q_network is the online Q-network (nn.Module), callable as
#         self.q_network(state_tensor) -> Tensor[batch, n_actions]
#   - self.device is a torch.device
#   - the method called every env step to pick an action is named
#         select_action(self, state) -> int
#     and currently implements epsilon-greedy over self.q_network.

# If your method is named act() / choose_action() / get_action() instead,
# or your network output isn't raw Q-values (e.g. it returns dueling
# streams already summed -- normally fine, just double check), rename
# the override below to match.
# --------------------------------------------------------------------
# """

# import numpy as np
# import torch
# import torch.nn.functional as F

# from dqn_agent import DQNAgent


# class BoltzmannDQNAgent(DQNAgent):
#     """
#     Same network / replay / update logic as DQNAgent. Only the
#     exploration policy changes: softmax over Q-values with a
#     temperature that anneals over training, instead of epsilon-greedy.

#         P(a | s) = exp(Q(s,a) / T) / sum_a' exp(Q(s,a') / T)

#     High T -> close to uniform random (exploration)
#     Low  T -> close to argmax / greedy (exploitation)

#     Unlike epsilon-greedy, there's no hard floor where exploration
#     suddenly stops -- the policy stays a smooth function of Q-value
#     gaps the whole time, which is the property you want to test given
#     that D3QN v1 diverged right when epsilon bottomed out.
#     """

#     def __init__(
#         self,
#         *args,
#         temp_start: float = 1.0,
#         temp_end: float = 0.2,
#         temp_decay_steps: int = 200_000,
#         temp_schedule: str = "linear",  # "linear" or "exponential"
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.temp_start = temp_start
#         self.temp_end = temp_end
#         self.temp_decay_steps = temp_decay_steps
#         self.temp_schedule = temp_schedule
#         self.train_step_count = 0  # increment once per env step (not per episode)

#     def current_temperature(self) -> float:
#         frac = min(1.0, self.train_step_count / max(1, self.temp_decay_steps))
#         if self.temp_schedule == "exponential":
#             # smooth exponential decay from temp_start to temp_end
#             ratio = self.temp_end / self.temp_start
#             return self.temp_start * (ratio ** frac)
#         # default: linear
#         return self.temp_start + frac * (self.temp_end - self.temp_start)

#     def set_step(self, step: int) -> None:
#         """
#         Optional: call this from your training loop if you'd rather
#         drive the temperature schedule off the outer loop's global
#         step counter (e.g. to keep it aligned with your logger) instead
#         of this class's internal counter.
#         """
#         self.train_step_count = step

#     @torch.no_grad()
#     def select_action(self, state):
#         """
#         Overrides epsilon-greedy select_action from DQNAgent.
#         `state` should match whatever shape/type your base class
#         expects (your 20-dim observation vector as a np.array).
#         """
#         state_t = torch.as_tensor(
#             state, dtype=torch.float32, device=self.device
#         ).unsqueeze(0)

#         q_values = self.q_network(state_t)  # shape [1, n_actions]

#         T = max(self.current_temperature(), 1e-6)  # avoid div-by-zero
#         probs = F.softmax(q_values / T, dim=-1).squeeze(0).cpu().numpy()

#         # Guard: extreme Q-value magnitudes can push softmax to a
#         # near-one-hot distribution even at moderate T (this is the
#         # main known failure mode of raw Boltzmann exploration -- watch
#         # for it especially early in training before Q-values settle).
#         if not np.all(np.isfinite(probs)):
#             probs = np.ones_like(probs) / len(probs)
#         else:
#             probs = probs / probs.sum()  # renormalize for float drift

#         action = np.random.choice(len(probs), p=probs)

#         self.train_step_count += 1
#         return int(action)

#     def action_distribution(self, state):
#         """
#         Debug helper: returns (probs, temperature) for a given state
#         without sampling or advancing the step counter. Useful for
#         sanity-checking that T is actually annealing and that probs
#         aren't collapsing to one-hot early on.
#         """
#         state_t = torch.as_tensor(
#             state, dtype=torch.float32, device=self.device
#         ).unsqueeze(0)
#         with torch.no_grad():
#             q_values = self.q_network(state_t)
#         T = max(self.current_temperature(), 1e-6)
#         probs = F.softmax(q_values / T, dim=-1).squeeze(0).cpu().numpy()
#         return probs, T

"""
boltzmann_agent.py

Standalone Boltzmann (softmax) exploration experiment.

Does NOT modify dqn_agent.py. This subclasses DQNAgent and overrides
ONLY the action-selection method, so the network architecture, replay
buffer, and training/update logic stay exactly as they are in your
working D3QN implementation.

--------------------------------------------------------------------
ASSUMPTIONS ABOUT YOUR DQNAgent (from dqn_agent.py) -- fix if wrong:
  - class DQNAgent(...) lives in dqn_agent.py
  - self.q_network is the online Q-network (nn.Module), callable as
        self.q_network(state_tensor) -> Tensor[batch, n_actions]
  - self.device is a torch.device
  - the method called every env step to pick an action is named
        select_action(self, state) -> int
    and currently implements epsilon-greedy over self.q_network.

If your method is named act() / choose_action() / get_action() instead,
or your network output isn't raw Q-values (e.g. it returns dueling
streams already summed -- normally fine, just double check), rename
the override below to match.
--------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F

from dqn_agent import DQNAgent


class BoltzmannDQNAgent(DQNAgent):
    """
    Same network / replay / update logic as DQNAgent. Only the
    exploration policy changes: softmax over Q-values with a
    temperature that anneals over training, instead of epsilon-greedy.

        P(a | s) = exp(Q(s,a) / T) / sum_a' exp(Q(s,a') / T)

    High T -> close to uniform random (exploration)
    Low  T -> close to argmax / greedy (exploitation)

    Unlike epsilon-greedy, there's no hard floor where exploration
    suddenly stops -- the policy stays a smooth function of Q-value
    gaps the whole time, which is the property you want to test given
    that D3QN v1 diverged right when epsilon bottomed out.
    """

    def __init__(
        self,
        *args,
        temp_start: float = 1.0,
        temp_end: float = 0.05,
        temp_decay_steps: int = 200_000,
        temp_schedule: str = "linear",  # "linear" or "exponential"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_decay_steps = temp_decay_steps
        self.temp_schedule = temp_schedule
        self.train_step_count = 0  # increment once per env step (not per episode)

    def current_temperature(self) -> float:
        frac = min(1.0, self.train_step_count / max(1, self.temp_decay_steps))
        if self.temp_schedule == "exponential":
            # smooth exponential decay from temp_start to temp_end
            ratio = self.temp_end / self.temp_start
            return self.temp_start * (ratio ** frac)
        # default: linear
        return self.temp_start + frac * (self.temp_end - self.temp_start)

    def set_step(self, step: int) -> None:
        """
        Optional: call this from your training loop if you'd rather
        drive the temperature schedule off the outer loop's global
        step counter (e.g. to keep it aligned with your logger) instead
        of this class's internal counter.
        """
        self.train_step_count = step

    @torch.no_grad()
    def select_action(self, state):
        """
        Overrides epsilon-greedy select_action from DQNAgent.
        `state` should match whatever shape/type your base class
        expects (your 20-dim observation vector as a np.array).
        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        q_values = self.q_network(state_t)  # shape [1, n_actions]

        T = max(self.current_temperature(), 1e-6)  # avoid div-by-zero
        probs = F.softmax(q_values / T, dim=-1).squeeze(0).cpu().numpy()

        # Guard: extreme Q-value magnitudes can push softmax to a
        # near-one-hot distribution even at moderate T (this is the
        # main known failure mode of raw Boltzmann exploration -- watch
        # for it especially early in training before Q-values settle).
        if not np.all(np.isfinite(probs)):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()  # renormalize for float drift

        action = np.random.choice(len(probs), p=probs)

        self.train_step_count += 1
        return int(action)

    @torch.no_grad()
    def select_action_greedy(self, state):
        """
        Pure argmax over Q-values — no temperature, no sampling.
        For evaluation only. Deliberately does NOT touch
        train_step_count, so calling this doesn't perturb the
        temperature schedule used by select_action() during training.
        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=-1).item())

    def action_distribution(self, state):
        """
        Debug helper: returns (probs, temperature) for a given state
        without sampling or advancing the step counter. Useful for
        sanity-checking that T is actually annealing and that probs
        aren't collapsing to one-hot early on.
        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        T = max(self.current_temperature(), 1e-6)
        probs = F.softmax(q_values / T, dim=-1).squeeze(0).cpu().numpy()
        return probs, T