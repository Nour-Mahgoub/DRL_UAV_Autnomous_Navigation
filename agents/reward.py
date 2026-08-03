import numpy as np


def compute_reward(prev_dist, curr_dist, collided, action_vector,
                    alpha=1.0, beta=10.0, lam=0.01):
    """
    Reward from the grant proposal:
        r_t = alpha * (||g_{t+1}|| - ||g_{t}||) - beta * collision - lambda * ||a_t||^2

   

    Args:
        prev_dist: distance to goal at time t, ||g_t||
        curr_dist: distance to goal at time t+1, ||g_{t+1}||
        collided: bool, whether a collision occurred this step
        action_vector: array-like, the action taken (e.g. [vx, vy, vz])
        alpha: weight on goal-progress term
        beta: weight on collision penalty
        lam: weight on action-magnitude penalty (encourages smooth motion)

    Returns:
        float reward for this step
    """
    progress_term = alpha * (prev_dist - curr_dist)
    collision_term = beta * float(collided)
    action_penalty = lam * float(np.sum(np.square(action_vector)))

    reward = progress_term - collision_term - action_penalty
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