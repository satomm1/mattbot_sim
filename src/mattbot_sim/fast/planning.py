"""Path planning as localize_and_navigate.replan() does it: A* (navigation_utils.search.AStar) on the
static navigation grid combined with the robot's object layer, then compute_smoothed_traj."""

import contextlib
import copy
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from mattbot_sim.fast import imports  # noqa: F401  (library paths)
from navigation_utils import compute_smoothed_traj, plan_start_heading
from navigation_utils.grids import StochOccupancyGrid2D
from navigation_utils.search import AStar

RES = 0.05  # navigator plan_resolution
CORNER = dict(corner_aware=True, corner_min_turn_deg=45.0, corner_inset_m=0.10, corner_points_per_leg=3)


@dataclass
class Plan:
    path: List[Tuple[float, float]]  # A* path (grid points)
    times: Optional[np.ndarray]  # trajectory sample times (s), None when too short to smooth
    traj: Optional[np.ndarray]  # (N, 7): x, y, th, xd, yd, xdd, ydd
    th_init: float = 0.0

    @property
    def short(self):
        """Fewer than 4 path points: the navigator parks (or arrives at a detour viewpoint) instead."""
        return self.traj is None


def snap(xy):
    return (round(xy[0] / RES) * RES, round(xy[1] / RES) * RES)


class Planner:
    """One per map (shared by all robots). occupancy: HxW int array (ROS layout rows = y)."""

    def __init__(self, occupancy, origin, resolution, robot_d, v_des, robots_d=0.4):
        h, w = occupancy.shape
        self.shape = (h, w)
        self.origin = (float(origin[0]), float(origin[1]))
        self.resolution = float(resolution)
        self.robot_d = float(robot_d)
        self.robots_d = float(robots_d)  # AStar: cells closer than this to another robot are blocked (its default)
        self.v_des = float(v_des)
        self.static = StochOccupancyGrid2D(self.resolution, w, h, self.origin[0], self.origin[1], 5,
                                           np.asarray(occupancy, dtype=np.int16).ravel(), robot_d=self.robot_d)
        self._layer_cache = {}

    def grid(self, object_layer=None, version=None):
        """Static grid, or the cell-wise max with an object layer (HxW, 100 = blocked) as the navigator's
        _planning_occupancy(). ``version`` lets repeated calls reuse the combined grid."""
        if object_layer is None:
            return self.static
        key = version if version is not None else id(object_layer)
        cached = self._layer_cache.get(key)
        if cached is None:
            cached = copy.copy(self.static)
            cached.probs = np.maximum(self.static.probs, np.asarray(object_layer, dtype=self.static.probs.dtype)
                                      .reshape(self.static.probs.shape))
            self._layer_cache = {key: cached}  # one live object-layer version per robot is enough
        return cached

    def object_grid(self, object_layer, robot_d):
        """Object layer alone (for path_still_valid), with its own robot_d."""
        g = copy.copy(self.static)
        g.probs = np.asarray(object_layer, dtype=self.static.probs.dtype).reshape(self.static.probs.shape)
        g.robot_d = robot_d
        return g

    def start_cell(self, grid, x, y):
        """Navigator _resolve_plan_start: snapped pose, or the nearest free cell within 1 m."""
        s = snap((x, y))
        if grid.is_free(s):
            return s
        found, _d = grid.find_nearest_free(s, max_radius=1.0, step=RES)
        return None if found is None else snap(found)

    def plan(self, start, goal, grid, robots=(), goal_snap_m=0.0):
        """A* start -> goal; robots: [(x, y)] of other robots to plan around. Returns Plan or None (no path).

        goal_snap_m > 0 moves a goal that is not free to the nearest free cell (detour viewpoints, 0.4 m)."""
        g = snap(goal)
        if not grid.is_free(g):
            if goal_snap_m <= 0:
                return None
            found, _d = grid.find_nearest_free(g, max_radius=goal_snap_m, step=RES)
            if found is None:
                return None
            g = snap(found)
        ext = grid.extent
        rx = [r[0] for r in robots] if robots else None
        ry = [r[1] for r in robots] if robots else None
        problem = AStar((ext[0], ext[2]), (ext[1], ext[3]), start, g, grid, RES,
                        robots_x=rx, robots_y=ry, obj_x=[] if robots else None, obj_y=[] if robots else None,
                        obj_d=[] if robots else None, robots_d=self.robots_d, max_plan_time_sec=60.0)
        with contextlib.redirect_stdout(io.StringIO()):  # AStar prints every solve
            ok = problem.solve()
        if not ok or problem.path is None:
            return None
        path = [tuple(map(float, p)) for p in problem.path]
        if len(path) < 4:
            return Plan(path, None, None)
        times, traj = compute_smoothed_traj(path, self.v_des, 3, 0.15, 0.1, **CORNER)
        return Plan(path, np.asarray(times), np.asarray(traj), float(plan_start_heading(path, traj, v_min=0.05)))
