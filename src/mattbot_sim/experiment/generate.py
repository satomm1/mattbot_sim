"""Instances of an experiment. Pure Python (no ROS).

Obstacle candidates are computed once per spec (and cached): points every candidate_step_m along the roadmap's
edges (corridor centre lines), at least clearance_m from every landmark and robot start, whose square footprint
(width_m) leaves every landmark reachable from every other (a blocked passage always has a way around) and whose
obstacle importance I_o over the landmark trips is at least min_importance_m.

An instance (one per seed, shared by every condition) holds the task list (destination landmarks released as a
Poisson process) and the obstacle schedule (candidates at least min_separation_m apart, each added at a uniform
time in appear_s and removed lifetime_s later). write_scenario() turns it into a fleet scenario YAML for FastSim.
"""

import hashlib
import json
import math
import os
from collections import OrderedDict

import numpy as np

from mattbot_sim.fast import imports  # noqa: F401
from mattbot_sim.detour_check import load_maps, occupancy_tuple

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ros", "mattbot_experiments")


class MapContext:
    """The map, the fast-sim planner grid (for goal checks) and the observation planner's roadmap."""

    def __init__(self, map_name, map_json_dir=None, robot_radius=0.4, roadmap_cache_dir=None):
        from mattbot_sim.fast.planning import Planner
        from navigation_utils.roadmap import RoadmapParams, load_or_build_roadmap

        self.map_name = map_name
        self.grid, grid_mod = load_maps(map_name, map_json_dir or imports.MAP_JSON_DIR)
        nav_occ = np.maximum(self.grid.occupancy, grid_mod.occupancy)
        self.planner = Planner(nav_occ, (self.grid.origin_x, self.grid.origin_y), self.grid.resolution,
                               2 * robot_radius, 0.4)
        self.occupancy = occupancy_tuple(self.grid)
        self.robot_radius = robot_radius
        self.roadmap, _cached = load_or_build_roadmap(*self.occupancy, params=RoadmapParams(robot_radius=robot_radius),
                                                      cache_dir=roadmap_cache_dir)

    def is_free(self, x, y):
        return bool(self.planner.grid().is_free((x, y)))

    def snap(self, x, y):
        """(x, y) if the planner can go there, else the nearest roadmap node."""
        if self.is_free(x, y):
            return float(x), float(y)
        n = self.roadmap.nearest_node(x, y)
        return tuple(float(v) for v in self.roadmap.node_xy[n])


def snap_landmarks(ctx, landmarks):
    return OrderedDict((name, ctx.snap(x, y) + (th,)) for name, (x, y, th) in landmarks.items())


def unreachable_landmarks(ctx, landmarks, from_xy):
    """Landmarks the fast-sim planner (the navigator's A* and footprint) cannot reach from from_xy on the static
    map. A task to one of them always fails, so the spec must not use them."""
    grid = ctx.planner.grid()
    start = ctx.planner.start_cell(grid, from_xy[0], from_xy[1])
    if start is None:
        raise ValueError("robot start (%.2f, %.2f) is not free" % tuple(from_xy))
    return [name for name, (x, y, _th) in landmarks.items() if ctx.planner.plan(start, (x, y), grid) is None]


# ---------- Obstacle candidates ----------

def edge_points(roadmap, step_m):
    """Points every step_m (arc length) along every roadmap edge polyline, world frame."""
    out = []
    for u, v in roadmap.graph.edges:
        poly = np.asarray(roadmap.graph.edges[u, v]["polyline"], dtype=float)
        xy = np.stack([roadmap.origin[0] + (poly[:, 1] + 0.5) * roadmap.res,
                       roadmap.origin[1] + (poly[:, 0] + 0.5) * roadmap.res], axis=1)
        if len(xy) < 2:
            continue
        seg = np.hypot(*np.diff(xy, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        for s in np.arange(step_m / 2.0, arc[-1], step_m):
            k = int(np.searchsorted(arc, s))
            out.append((float(xy[min(k, len(xy) - 1), 0]), float(xy[min(k, len(xy) - 1), 1])))
    return out


class Connectivity:
    """Do square footprints cut a landmark off? Labels the roadmap's C-space with the footprints blocked, each
    inflated by margin_m: the planner's square robot (robot_d = 2 x robot_radius) reaches a corner at
    robot_radius x sqrt(2), so that is the default."""

    def __init__(self, roadmap, landmark_xy, robot_radius, margin_m=None):
        self.rm = roadmap
        self.margin_m = robot_radius * math.sqrt(2.0) if margin_m is None else margin_m
        self.cells = [roadmap.world_to_cell(x, y) for x, y in landmark_xy]

    def keeps_connected(self, x, y, width):
        return self.keeps_connected_all([(x, y, width)])

    def keeps_connected_all(self, footprints):
        """All footprints [(x, y, width)] in place at once."""
        from scipy import ndimage

        rm = self.rm
        free = rm.cfree.copy()
        for x, y, width in footprints:
            half = width / 2.0 + self.margin_m
            r0, c0 = rm.world_to_cell(x - half, y - half)
            r1, c1 = rm.world_to_cell(x + half, y + half)
            free[max(r0, 0):max(r1 + 1, 0), max(c0, 0):max(c1 + 1, 0)] = False
        labels, _n = ndimage.label(free, structure=np.ones((3, 3), dtype=bool))
        seen = set()
        for r, c in self.cells:
            if not (0 <= r < free.shape[0] and 0 <= c < free.shape[1]):
                return False
            seen.add(int(labels[r, c]))
        return len(seen) == 1 and 0 not in seen


def _candidates_key(spec, ctx, landmarks):
    o = spec.obstacles
    h = hashlib.sha256()
    h.update(json.dumps({"map": spec.map, "roadmap": ctx.roadmap.fingerprint, "landmarks": list(landmarks.items()),
                         "starts": [r["start"] for r in spec.robots], "width": o["width_m"],
                         "step": o["candidate_step_m"], "clearance": o["clearance_m"], "version": 2},
                        sort_keys=True).encode())
    return h.hexdigest()[:16]


def compute_candidates(spec, ctx, landmarks, cache_dir=DEFAULT_CACHE_DIR, progress=None):
    """[{x, y, I_o, frac_affected}] of every usable placement (any importance; filter with min_importance_m)."""
    from navigation_utils.edge_blocking import BlockingParams
    from navigation_utils.obstacle_importance import (
        UNIFORM_LANDMARKS, ImportanceEvaluator, ImportanceParams, build_trip_set,
    )

    o = spec.obstacles
    key = _candidates_key(spec, ctx, landmarks)
    path = os.path.join(cache_dir, "candidates_%s.json" % key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    rm = ctx.roadmap
    keep_away = [(x, y) for x, y, _th in landmarks.values()] + [r["start"][:2] for r in spec.robots]
    lm_file = os.path.join(cache_dir, "landmarks_%s.yaml" % key)
    os.makedirs(cache_dir, exist_ok=True)
    with open(lm_file, "w") as f:  # the snapped landmarks, for the trip set
        f.write("".join("%s: [%f, %f]\n" % (n, x, y) for n, (x, y, _th) in landmarks.items()))
    iparams = ImportanceParams(trip_mode=UNIFORM_LANDMARKS, landmarks_file=lm_file)
    evaluator = ImportanceEvaluator(rm, build_trip_set(rm, iparams), iparams, BlockingParams(), cache_dir=cache_dir)
    conn = Connectivity(rm, [(x, y) for x, y, _th in landmarks.values()], ctx.robot_radius)
    points = [(x, y) for x, y in edge_points(rm, float(o["candidate_step_m"]))
              if min(math.hypot(x - a, y - b) for a, b in keep_away) >= float(o["clearance_m"]) and ctx.is_free(x, y)]
    out = []
    for k, (x, y) in enumerate(points):
        if progress is not None:
            progress(k, len(points))
        if not conn.keeps_connected(x, y, float(o["width_m"])):
            continue  # would cut a landmark off: no way around
        r = evaluator.evaluate_square("cand_%d" % k, x, y, float(o["width_m"]))
        out.append({"x": round(x, 3), "y": round(y, 3), "I_o": round(float(r.I_o), 3),
                    "frac_affected": round(float(r.frac_affected), 4)})
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def setup(spec, cache_dir=DEFAULT_CACHE_DIR, progress=None):
    """(ctx, landmarks, candidates, connectivity) of a spec. Raises if a landmark is unreachable."""
    ctx = MapContext(spec.map)
    landmarks = snap_landmarks(ctx, spec.landmarks)
    bad = unreachable_landmarks(ctx, landmarks, spec.robots[0]["start"][:2])
    if bad:
        raise ValueError("%s: the planner cannot reach landmark(s) %s; move them in %s" % (
            spec.path, ", ".join(bad), spec.landmarks_path))
    candidates = compute_candidates(spec, ctx, landmarks, cache_dir, progress)
    conn = Connectivity(ctx.roadmap, [(x, y) for x, y, _th in landmarks.values()], ctx.robot_radius)
    return ctx, landmarks, candidates, conn


# ---------- Instances ----------

def make_instance(spec, seed, candidates, landmarks, connectivity=None):
    """Task list and obstacle schedule of one seed (the same for every condition). With connectivity
    (a Connectivity), the obstacles are also chosen so that all of them together cut no landmark off: each
    one alone leaves a way around, but two can close the same region."""
    t, o = spec.tasks, spec.obstacles
    rng = np.random.default_rng([int(seed), 20261020])
    names = list(landmarks)
    tasks, now, prev = [], float(t["start_s"]), None
    rate = float(t["release_rate_per_min"])
    for i in range(int(t["count"])):
        now += float(rng.exponential(60.0 / rate)) if rate > 0 else 0.0
        name = str(rng.choice([n for n in names if n != prev]))
        prev = name
        x, y, th = landmarks[name]
        tasks.append({"id": i, "release_s": round(now, 2), "landmark": name, "x": x, "y": y, "theta": th})

    usable = [c for c in candidates if c["I_o"] >= float(o["min_importance_m"])]
    order = rng.permutation(len(usable))
    chosen = []
    for k in order:
        c = usable[int(k)]
        if any(math.hypot(c["x"] - d["x"], c["y"] - d["y"]) < float(o["min_separation_m"]) for d in chosen):
            continue
        if connectivity is not None and not connectivity.keeps_connected_all(
                [(d["x"], d["y"], float(o["width_m"])) for d in chosen + [c]]):
            continue
        chosen.append(c)
        if len(chosen) >= int(o["count"]):
            break
    obstacles, events = [], []
    a0, a1 = (float(v) for v in o["appear_s"])
    l0, l1 = (float(v) for v in o["lifetime_s"])
    classes = list(o["classes"])
    for i, c in enumerate(chosen):
        appear = round(float(rng.uniform(a0, a1)), 1)
        remove = round(appear + float(rng.uniform(l0, l1)), 1)
        ob = {"id": "obs_%d" % i, "class": str(rng.choice(classes)), "x": c["x"], "y": c["y"],
              "width": float(o["width_m"]), "I_o": c["I_o"], "appear_s": appear, "remove_s": remove}
        obstacles.append(ob)
        events.append({"at": appear, "add": {k: ob[k] for k in ("id", "class", "x", "y", "width")}})
        events.append({"at": remove, "remove": ob["id"]})
    events.sort(key=lambda e: e["at"])
    return {"spec": spec.name, "seed": int(seed), "tasks": tasks, "obstacles": obstacles, "events": events,
            "candidates_usable": len(usable)}


def write_scenario(spec, instance, path):
    """Fleet scenario YAML for FastSim: the robots (no waypoints: the task orchestrator drives them), no objects
    at the start, the obstacle schedule as events."""
    import yaml

    d = {"name": "%s_seed%d" % (spec.name, instance["seed"]), "map": spec.map, "duration": spec.max_duration_s,
         "objects": [], "events": instance["events"],
         "robots": [{"id": r["id"], "start": {"x": r["start"][0], "y": r["start"][1], "theta": r["start"][2]},
                     "loop": False} for r in spec.robots]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    return path


def plot_instance(spec, ctx, landmarks, candidates, instance, path):
    """PNG for review: map, landmarks, every candidate (colour = I_o), the chosen obstacles and robot starts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, w, h, res, origin = ctx.occupancy
    occ = np.asarray(data).reshape(h, w)
    img = np.where((occ >= 0) & (occ < 50), 1.0, np.where(occ < 0, 0.6, 0.0))
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1,
              extent=(origin[0], origin[0] + w * res, origin[1], origin[1] + h * res))
    if candidates:
        cx = [c["x"] for c in candidates]
        cy = [c["y"] for c in candidates]
        sc = ax.scatter(cx, cy, c=[c["I_o"] for c in candidates], s=10, cmap="viridis", zorder=2)
        fig.colorbar(sc, ax=ax, fraction=0.02, label="I_o (m per trip)")
    for name, (x, y, _th) in landmarks.items():
        ax.plot(x, y, "b^", ms=8, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color="b")
    for r in spec.robots:
        ax.plot(r["start"][0], r["start"][1], "go", ms=8, zorder=3)
        ax.annotate("robot %d" % r["id"], r["start"][:2], textcoords="offset points", xytext=(4, -10), fontsize=8,
                    color="g")
    for ob in instance["obstacles"]:
        hw = ob["width"] / 2.0
        ax.add_patch(plt.Rectangle((ob["x"] - hw, ob["y"] - hw), ob["width"], ob["width"], color="r", zorder=4))
        ax.annotate("%s I_o=%.1f\n%.0f-%.0f s" % (ob["id"], ob["I_o"], ob["appear_s"], ob["remove_s"]),
                    (ob["x"], ob["y"]), textcoords="offset points", xytext=(6, 6), fontsize=7, color="r")
    free = np.nonzero((occ >= 0) & (occ < 50))
    ax.set_xlim(origin[0] + free[1].min() * res - 1, origin[0] + free[1].max() * res + 1)
    ax.set_ylim(origin[1] + free[0].min() * res - 1, origin[1] + free[0].max() * res + 1)
    ax.set_title("%s seed %d: %d tasks, %d obstacles (%d usable candidates with I_o >= %g m)" % (
        spec.name, instance["seed"], len(instance["tasks"]), len(instance["obstacles"]),
        instance["candidates_usable"], spec.obstacles["min_importance_m"]))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
