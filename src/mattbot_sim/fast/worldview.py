"""The shared world plus the robots in it: other robots block line of sight (they are not detected)."""

import numpy as np

from mattbot_sim.world import World, ray_circle


class FleetWorld(World):
    """World whose raycasts also hit the robots' discs (except the observer's own)."""

    def __init__(self, grid, objects=(), now=0.0):
        super().__init__(grid, objects, now)
        self.robot_discs = {}  # robot_id -> (x, y, radius)
        self.observer = None  # robot_id whose camera is looking (excluded from raycasts)

    def raycast(self, x, y, angles, max_range, include_objects=True, exclude=()):
        ranges = super().raycast(x, y, angles, max_range, include_objects, exclude)
        if include_objects and self.robot_discs:
            angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
            for rid, (rx, ry, r) in self.robot_discs.items():
                if rid != self.observer:
                    ranges = np.minimum(ranges, ray_circle(x, y, angles, rx, ry, r))
            ranges[ranges > max_range] = np.inf
        return ranges
