import numpy as np


def compute_reward(prev_dist, curr_dist, collided, action_vector,
                    reached_goal=False,
                    alpha=1.0, beta=10.0, lam=0.01, success_bonus=50.0):
    """
    Reward from the grant proposal, extended with a terminal success bonus:
        r_t = alpha * (||g_t|| - ||g_{t+1}||) - beta * collision - lambda * ||a_t||^2
              + success_bonus * reached_goal

    The first three terms are unchanged from the original formulation. The
    success_bonus term is new: without it, a policy that approaches the goal
    and hovers nearby accumulates almost the same reward as one that actually
    reaches it and stops — nothing in the signal marks "arrived" as
    categorically different from "very close." success_bonus is a one-time,
    large, sparse spike awarded only on the single step where the drone
    first enters accept_radius, meant to dominate the accumulated per-step
    progress reward so that finishing is unambiguously better than almost
    finishing.

    Args:
        prev_dist: distance to goal at time t, ||g_t||
        curr_dist: distance to goal at time t+1, ||g_{t+1}||
        collided: bool, whether a collision occurred this step
        action_vector: array-like, the action taken (e.g. [vx, vy, vz])
        reached_goal: bool, whether curr_dist < accept_radius this step —
            pass this in from the caller (airsim_env.py already computes it
            for the done/info logic; just reuse that value here rather than
            recomputing accept_radius comparisons in two places)
        alpha: weight on goal-progress term
        beta: weight on collision penalty
        lam: weight on action-magnitude penalty (encourages smooth motion)
        success_bonus: one-time reward awarded on the step goal is reached.
            Sized to comfortably exceed the total progress reward achievable
            over a typical episode (roughly the start-to-goal distance, since
            the progress term telescopes to that sum) — default 50.0 assumes
            a ~20m nominal flight, so tune upward if using much larger goal
            distances or jitter radii.

    Returns:
        float reward for this step
    """
    progress_term = alpha * (prev_dist - curr_dist)
    collision_term = beta * float(collided)
    action_penalty = lam * float(np.sum(np.square(action_vector)))
    bonus_term = success_bonus * float(reached_goal)

    reward = progress_term - collision_term - action_penalty + bonus_term
    return reward


if __name__ == "__main__":
    # quick manual sanity checks
    # case 1: moved closer to goal, no collision, small action -> should be positive-ish
    r1 = compute_reward(prev_dist=10.0, curr_dist=9.0, collided=False,
                         action_vector=[2.0, 0.0, 0.0])
    print(f"Moved closer, no collision: reward = {r1:.3f} (expect positive)")

    # case 2: moved away from goal -> should be negative
    r2 = compute_reward(prev_dist=9.0, curr_dist=10.0, collided=False,
                         action_vector=[2.0, 0.0, 0.0])
    print(f"Moved away, no collision: reward = {r2:.3f} (expect negative)")

    # case 3: collision -> should be strongly negative regardless of progress
    r3 = compute_reward(prev_dist=10.0, curr_dist=9.0, collided=True,
                         action_vector=[2.0, 0.0, 0.0])
    print(f"Moved closer but collided: reward = {r3:.3f} (expect negative, dominated by collision penalty)")

    # case 4: reached goal this step -> should be large and clearly positive,
    # bigger than any single-step progress reward could plausibly be
    r4 = compute_reward(prev_dist=2.5, curr_dist=1.8, collided=False,
                         action_vector=[1.0, 0.0, 0.0], reached_goal=True)
    print(f"Reached goal: reward = {r4:.3f} (expect large positive, ~50+)")

    # case 5: hovering near goal WITHOUT crossing accept_radius, repeated —
    # this is the exact pattern the bonus is meant to out-compete. Simulate
    # 20 steps of tiny back-and-forth jitter near the goal, no success.
    hover_total = sum(
        compute_reward(prev_dist=2.2, curr_dist=2.15, collided=False,
                        action_vector=[0.3, 0.0, 0.0])
        for _ in range(20)
    )
    print(f"20 steps hovering near goal, never reaching it: total reward = {hover_total:.3f}")
    print(f"Single successful arrival: reward = {r4:.3f}")
    print("Arrival should clearly beat sustained hovering — compare the two lines above.")