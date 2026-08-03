import itertools


class ActionSpace:
    """
    27 discrete actions = 3 (x) x 3 (y) x 3 (z), each axis in {-1, 0, +1}.
    Index 13 (0,0,0) = hover.
    """

    def __init__(self, speed=2.0):
        self.speed = speed
        levels = [-1, 0, 1]
        # itertools.product gives all 27 combinations in a fixed, deterministic order
        self.action_table = list(itertools.product(levels, levels, levels))
        self.n = len(self.action_table)  # 27

        # sanity check — confirm hover is where we expect it
        self.hover_idx = self.action_table.index((0, 0, 0))

    def decode(self, action_idx):
        """
        action_idx (int, 0-26) -> (vx, vy, vz) in m/s, world/NED frame.
        """
        vx_dir, vy_dir, vz_dir = self.action_table[action_idx]
        return (vx_dir * self.speed, vy_dir * self.speed, vz_dir * self.speed)

    def describe(self, action_idx):
        """Human-readable label, useful for debugging/logging."""
        vx_dir, vy_dir, vz_dir = self.action_table[action_idx]
        x_label = {1: "forward", 0: "", -1: "backward"}[vx_dir]
        y_label = {1: "right", 0: "", -1: "left"}[vy_dir]
        z_label = {1: "down", 0: "", -1: "up"}[vz_dir]  # NED: +z is down
        parts = [p for p in (x_label, y_label, z_label) if p]
        return "+".join(parts) if parts else "hover"


if __name__ == "__main__":
    # quick manual check — run this file directly to confirm the mapping looks right
    action_space = ActionSpace(speed=2.0)
    print(f"Total actions: {action_space.n}")
    print(f"Hover index: {action_space.hover_idx}")
    for i in range(action_space.n):
        print(i, action_space.decode(i), "->", action_space.describe(i))