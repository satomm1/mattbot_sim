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

Optional:
    extends: main.yaml          # start from another spec (relative to this one); sections merge key by key
    end_at_horizon: true        # always run to max_duration_s (default: end when every task is done or failed)
    tasks: {release_rate_per_robot_per_min: 0.17, until_s: 3600}   # rate x number of robots; release until
                                                                    # until_s (in place of count)
    obstacles:
      types:                    # typed obstacles (in place of count / min_importance_m / width_m / classes)
        blocking: {count: 2, width_m: 1.0, classes: [bench], min_importance_m: 1.0}
        on_route: {count: 2, width_m: 0.5, classes: [chair, cone]}
        visible:  {count: 2, width_m: 0.5, classes: [chair, cone], max_importance_m: 0.5}
      route_min_trip_share: 0.1 # an edge is on a frequent route if this share of landmark trips use it
      off_route_min_m: 1.5      # visible obstacles are at least this far from every frequent-route cell

Obstacle types (see generate.compute_candidates): blocking blocks a roadmap edge (a way around remains) and has
I_o >= min_importance_m; on_route sits on a frequent-route edge and blocks none (robots pass it); visible is off
the frequent routes but in view (observation range, line of sight) of one.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import yaml

TASK_DEFAULTS = {"count": 40, "release_rate_per_min": 2.0, "dwell_s": 5.0, "start_s": 0.0}
OBSTACLE_DEFAULTS = {"count": 4, "min_importance_m": 1.0, "width_m": 1.0, "classes": ["chair"],
                     "appear_s": [0.0, 900.0], "lifetime_s": [300.0, 1500.0], "min_separation_m": 3.0,
                     "clearance_m": 2.0, "candidate_step_m": 1.0, "route_min_trip_share": 0.1,
                     "off_route_min_m": 1.5}
OBSTACLE_TYPES = ("blocking", "on_route", "visible")  # sampling order in an instance
TYPE_DEFAULTS = {"count": 0, "width_m": 0.5, "classes": ["chair"], "min_importance_m": 0.0, "max_importance_m": None}


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
    end_at_horizon: bool = False
    path: str = ""  # the spec file
    landmarks_path: str = ""
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def obstacle_types(self):
        """{type: {count, width_m, classes, min_importance_m}} in OBSTACLE_TYPES order (flat keys: blocking)."""
        return self.obstacles["types"]

    @property
    def release_rate_per_min(self):
        t = self.tasks
        if t.get("release_rate_per_robot_per_min") is not None:
            return float(t["release_rate_per_robot_per_min"]) * len(self.robots)
        return float(t["release_rate_per_min"])

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


def merge(base, over):
    """over on top of base: dicts merge key by key (recursively), anything else is replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_spec_dict(path, _seen=()):
    """The spec file as a dict, with ``extends`` resolved (relative paths: relative to the extending file)."""
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError("%s: circular extends" % path)
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("extends", None)
    if base is None:
        return d
    if not os.path.isabs(base):
        base = os.path.join(os.path.dirname(path), base)
    parent = read_spec_dict(base, _seen + (path,))
    lm = parent.get("landmarks")
    if lm and "landmarks" not in d and not os.path.isabs(lm):  # keep the base's landmarks file findable
        parent["landmarks"] = os.path.relpath(os.path.join(os.path.dirname(base), lm), os.path.dirname(path))
    if "robots" in d:  # a list: replaced, not merged
        parent.pop("robots", None)
    return merge(parent, d)


def _obstacle_types(o, path):
    """Typed obstacles, or the flat keys as one blocking type."""
    types = o.get("types")
    if not types:
        types = {"blocking": {k: o[k] for k in ("count", "width_m", "classes", "min_importance_m")}}
    unknown = set(types) - set(OBSTACLE_TYPES)
    if unknown:
        raise ValueError("%s: unknown obstacle type(s) %s (known: %s)" % (path, ", ".join(sorted(unknown)),
                                                                          ", ".join(OBSTACLE_TYPES)))
    return OrderedDict((t, dict(TYPE_DEFAULTS, **(types[t] or {}))) for t in OBSTACLE_TYPES if t in types)


def load_spec(path):
    path = os.path.abspath(path)
    d = read_spec_dict(path)
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
    obstacles["types"] = _obstacle_types(obstacles, path)
    seeds = d.get("seeds", 10)
    seeds = list(range(int(seeds))) if isinstance(seeds, int) else [int(s) for s in seeds]
    conditions = OrderedDict((str(k), dict(v or {})) for k, v in d["conditions"].items())
    known = {"name", "map", "landmarks", "robots", "tasks", "obstacles", "max_duration_s", "common", "conditions",
             "seeds", "end_at_horizon"}
    return ExperimentSpec(
        name=str(d["name"]), map=str(d["map"]), landmarks=load_landmarks(lm_path), robots=robots, tasks=tasks,
        obstacles=obstacles, max_duration_s=float(d.get("max_duration_s", 7200.0)), common=dict(d.get("common") or {}),
        conditions=conditions, seeds=seeds, end_at_horizon=bool(d.get("end_at_horizon", False)), path=path,
        landmarks_path=lm_path,
        extra={k: v for k, v in d.items() if k not in known},
    )
