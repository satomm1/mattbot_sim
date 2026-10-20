"""Experiment spec (YAML): map, landmarks, robots, task workload, obstacle schedule, conditions and seeds.

    name: pilot
    map: current_map
    landmarks: landmarks_current_map.yaml      # {name: [x, y(, theta)]}, relative to the spec; task destinations
    robots: [{id: 91, start: {x: 28.0, y: 18.95, theta: 0.0}}, ...]
    tasks: {count: 40, release_rate_per_min: 2.0, dwell_s: 5}
    obstacles: {count: 4, min_importance_m: 1.0, width_m: 1.0, classes: [chair], appear_s: [0, 900],
                lifetime_s: [300, 1500], min_separation_m: 3.0, clearance_m: 2.0, candidate_step_m: 1.0}
    max_duration_s: 7200
    common: {belief_hold_s: 60, ...}           # FastParams for every condition
    conditions: {none: {observe: false}, ...}   # FastParams per condition (over common)
    seeds: 20                                   # seeds 0 .. seeds-1 (or a list)
"""

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import yaml

TASK_DEFAULTS = {"count": 40, "release_rate_per_min": 2.0, "dwell_s": 5.0, "start_s": 0.0}
OBSTACLE_DEFAULTS = {"count": 4, "min_importance_m": 1.0, "width_m": 1.0, "classes": ["chair"],
                     "appear_s": [0.0, 900.0], "lifetime_s": [300.0, 1500.0], "min_separation_m": 3.0,
                     "clearance_m": 2.0, "candidate_step_m": 1.0}


@dataclass
class ExperimentSpec:
    name: str
    map: str
    landmarks: "OrderedDict[str, Tuple[float, float, float]]"
    robots: List[dict]  # [{"id": int, "start": (x, y, theta)}]
    tasks: Dict[str, float]
    obstacles: Dict[str, object]
    max_duration_s: float
    common: Dict[str, object]
    conditions: "OrderedDict[str, Dict[str, object]]"
    seeds: List[int]
    path: str = ""  # the spec file
    landmarks_path: str = ""
    extra: Dict[str, object] = field(default_factory=dict)

    def params_of(self, condition):
        """FastParams settings of one condition: common, then the condition's own."""
        out = dict(self.common)
        out.update(self.conditions[condition])
        return out


def _pose(d, what):
    if isinstance(d, dict):
        return float(d["x"]), float(d["y"]), float(d.get("theta", 0.0))
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        return float(d[0]), float(d[1]), float(d[2]) if len(d) > 2 else 0.0
    raise ValueError("%s: expected {x, y, theta} or [x, y, theta], got %r" % (what, d))


def load_landmarks(path):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and "landmarks" in data:
        data = data["landmarks"]
    out = OrderedDict((str(k), _pose(v, "landmark %s" % k)) for k, v in data.items())
    if len(out) < 2:
        raise ValueError("%s: need at least 2 landmarks" % path)
    return out


def load_spec(path):
    path = os.path.abspath(path)
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    for key in ("name", "map", "landmarks", "robots", "conditions"):
        if key not in d:
            raise ValueError("%s: missing %r" % (path, key))
    lm_path = d["landmarks"]
    if not os.path.isabs(lm_path):
        lm_path = os.path.join(os.path.dirname(path), lm_path)
    robots = [{"id": int(r["id"]), "start": _pose(r["start"], "robot start")} for r in d["robots"]]
    if len({r["id"] for r in robots}) != len(robots):
        raise ValueError("%s: duplicate robot ids" % path)
    tasks = dict(TASK_DEFAULTS, **(d.get("tasks") or {}))
    obstacles = dict(OBSTACLE_DEFAULTS, **(d.get("obstacles") or {}))
    seeds = d.get("seeds", 10)
    seeds = list(range(int(seeds))) if isinstance(seeds, int) else [int(s) for s in seeds]
    conditions = OrderedDict((str(k), dict(v or {})) for k, v in d["conditions"].items())
    known = {"name", "map", "landmarks", "robots", "tasks", "obstacles", "max_duration_s", "common", "conditions",
             "seeds"}
    return ExperimentSpec(
        name=str(d["name"]), map=str(d["map"]), landmarks=load_landmarks(lm_path), robots=robots, tasks=tasks,
        obstacles=obstacles, max_duration_s=float(d.get("max_duration_s", 7200.0)), common=dict(d.get("common") or {}),
        conditions=conditions, seeds=seeds, path=path, landmarks_path=lm_path,
        extra={k: v for k, v in d.items() if k not in known},
    )
