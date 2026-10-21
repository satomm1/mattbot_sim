"""Instances of an experiment. Pure Python (no ROS).

Obstacle candidates are computed once per spec (and cached), per obstacle type (spec.OBSTACLE_TYPES), at least
clearance_m from every landmark and robot start, with a square footprint (the type's width_m) that leaves every
landmark reachable from every other:
  blocking  points every candidate_step_m along the roadmap's edges (corridor centre lines) whose footprint blocks
            an edge (a way around remains); an instance uses those with I_o >= min_importance_m.
  on_route  beside frequent-route edges (used by >= route_min_trip_share of the landmark trips' shortest paths):
            from each edge point, sideways to the last free-floor spot before the wall (an object left at the side
            of a corridor), on both sides; it must block no edge, and the planner must drive past it (ledger
            blockout in place). Centred on the line, a 0.5 m object closes a 2 m corridor to the planner.
  visible   free floor on a VISIBLE_STEP_M grid, at least off_route_min_m from every frequent-route cell (in a room,
            an alcove or a side corridor; it may block a rarely used edge there: an instance keeps those with
            I_o <= the type's max_importance_m), in view of a frequent route by the observation planner's rule
            (viewpoints.width_line_of_sight_counts), on at least 1 m of frequent-route cells each way:
              discoverable: within the detector's range (r_min .. camera max_range) and within VIEW_AHEAD_DEG of the
                            direction of travel (either way along the route): the camera looks ahead, so an object
                            seen only sideways is rarely discovered;
              checkable:    within the observation range (r_min .. r_max), any direction (the robot turns to look).

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

def edge_samples(roadmap, step_m):
    """Points every step_m (arc length) along every roadmap edge polyline, world frame:
    [(x, y, edge, k, xy)] with edge = (min node, max node), k the polyline index and xy the edge's polyline."""
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
            k = min(int(np.searchsorted(arc, s)), len(xy) - 1)
            out.append((float(xy[k, 0]), float(xy[k, 1]), (min(u, v), max(u, v)), k, xy))
    return out


def edge_points(roadmap, step_m):
    """Points every step_m (arc length) along every roadmap edge polyline, world frame."""
    return [(x, y) for x, y, _e, _k, _xy in edge_samples(roadmap, step_m)]


def route_shares(evaluator):
    """{(min node, max node): share of the trips (by weight) whose obstacle-free shortest path uses that edge}."""
    shares = {}
    for s, t, w in zip(evaluator.s_nodes, evaluator.t_nodes, evaluator.trips.weights):
        if s == t:
            continue
        path = evaluator.table.path(int(s), int(t))
        if path is None:
            continue
        for a, b in zip(path[:-1], path[1:]):
            e = (min(a, b), max(a, b))
            shares[e] = shares.get(e, 0.0) + float(w)
    return shares


def edge_cells_xy(roadmap, edges, with_heading=False):
    """World (x, y) of the polyline cells of the given edges, (N, 2); with_heading also (N,) the polyline's
    direction there (rad, one of the two travel directions)."""
    pts, hdg = [], []
    for u, v in edges:
        poly = np.asarray(roadmap.graph.edges[u, v]["polyline"], dtype=float)
        xy = np.stack([roadmap.origin[0] + (poly[:, 1] + 0.5) * roadmap.res,
                       roadmap.origin[1] + (poly[:, 0] + 0.5) * roadmap.res], axis=1)
        pts.append(xy)
        if len(xy) > 1:
            nxt = np.concatenate([xy[1:], xy[-1:]])
            prv = np.concatenate([xy[:1], xy[:-1]])
            d = nxt - prv
            hdg.append(np.arctan2(d[:, 1], d[:, 0]))
        else:
            hdg.append(np.zeros(len(xy)))
    out = np.concatenate(pts) if pts else np.zeros((0, 2))
    if with_heading:
        return out, (np.concatenate(hdg) if hdg else np.zeros(0))
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
                         "starts": [r["start"] for r in spec.robots],
                         "widths": {t: v["width_m"] for t, v in spec.obstacle_types.items()},
                         "step": o["candidate_step_m"], "clearance": o["clearance_m"],
                         "route_share": o["route_min_trip_share"], "off_route": o["off_route_min_m"], "version": 7},
                        sort_keys=True).encode())
    return h.hexdigest()[:16]


VISIBLE_STEP_M = 0.5  # grid of visible candidates
VIEW_AHEAD_DEG = 25.0  # visible: seen within this of the route's direction (inside the camera's ~60 deg view)
SIDE_STEP_M = 0.1  # on_route: sideways search step from the corridor centre line
PASS_CHECK_M = 2.5  # on_route: the planner must get from this far before the obstacle to this far after it
PASS_DETOUR_M = 1.0  # ... with at most this much more than along the edge


def _passable(ctx, x, y, width, k, xy):
    """Can the planner drive along the edge past an obstacle at (x, y) (as a ledger blockout)?"""
    from navigation_utils.object_layers import ledger_blockout_grid

    seg = np.hypot(*np.diff(xy, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    a = int(np.searchsorted(arc, max(arc[k] - PASS_CHECK_M, 0.0)))
    b = min(int(np.searchsorted(arc, min(arc[k] + PASS_CHECK_M, arc[-1]))), len(xy) - 1)
    pl = ctx.planner
    h, w = pl.shape
    layer = ledger_blockout_grid([(x, y, width, 1.0)], w, h, pl.resolution, pl.origin)
    grid = pl.grid(layer, version=("pass", x, y, width))  # (the mapper's ledger blockout, 100 = blocked)
    start = pl.start_cell(grid, xy[a, 0], xy[a, 1])
    if start is None:
        return False
    plan = pl.plan(start, (xy[b, 0], xy[b, 1]), grid, goal_snap_m=0.4)
    if plan is None:
        return False
    length = float(np.sum(np.hypot(*np.diff(np.asarray(plan.path), axis=0).T)))
    return length <= (arc[b] - arc[a]) + PASS_DETOUR_M


def _against_wall(grid_occ, origin, res, x, y, k, xy, width, side, max_m=3.0):
    """From the edge point (x, y), sideways (side = +1 left / -1 right of the polyline's direction) to the last
    spot whose footprint is free floor; None if the first step already is not free."""
    a, b = xy[max(k - 1, 0)], xy[min(k + 1, len(xy) - 1)]
    tx, ty = b[0] - a[0], b[1] - a[1]
    n = math.hypot(tx, ty)
    if n < 1e-9:
        return None
    nx, ny = -ty / n * side, tx / n * side
    last = None
    for d in np.arange(SIDE_STEP_M, max_m + 1e-9, SIDE_STEP_M):
        px, py = x + d * nx, y + d * ny
        if not _free_footprint(grid_occ, origin, res, px, py, width):
            break
        last = (float(px), float(py), float(d))
    return last


def _free_footprint(grid_occ, origin, res, x, y, width):
    """Every static-map cell under the square footprint is known free floor."""
    h, w = grid_occ.shape
    half = width / 2.0
    c0, c1 = int(math.floor((x - half - origin[0]) / res)), int(math.floor((x + half - origin[0]) / res))
    r0, r1 = int(math.floor((y - half - origin[1]) / res)), int(math.floor((y + half - origin[1]) / res))
    if c0 < 0 or r0 < 0 or c1 >= w or r1 >= h:
        return False
    cells = grid_occ[r0:r1 + 1, c0:c1 + 1]
    return bool(np.all((cells >= 0) & (cells < 50)))


def compute_candidates(spec, ctx, landmarks, cache_dir=DEFAULT_CACHE_DIR, progress=None):
    """[{type, x, y, width, I_o, frac_affected, blocked, route_share}] of every usable placement of every obstacle
    type in the spec (blocking candidates at any importance; filter with min_importance_m)."""
    from navigation_utils.edge_blocking import BlockingParams
    from navigation_utils.obstacle_importance import (
        UNIFORM_LANDMARKS, ImportanceEvaluator, ImportanceParams, build_trip_set,
    )
    from navigation_utils.viewpoints import LOS_MIN_CLEAR, blocking_mask, width_line_of_sight_counts
    from mattbot_sim.fast.params import FastParams
    from scipy import ndimage

    o = spec.obstacles
    types = spec.obstacle_types
    key = _candidates_key(spec, ctx, landmarks)
    path = os.path.join(cache_dir, "candidates_%s.json" % key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    rm = ctx.roadmap
    keep_away = [(x, y) for x, y, _th in landmarks.values()] + [r["start"][:2] for r in spec.robots]
    clearance = float(o["clearance_m"])
    lm_file = os.path.join(cache_dir, "landmarks_%s.yaml" % key)
    os.makedirs(cache_dir, exist_ok=True)
    with open(lm_file, "w") as f:  # the snapped landmarks, for the trip set
        f.write("".join("%s: [%f, %f]\n" % (n, x, y) for n, (x, y, _th) in landmarks.items()))
    iparams = ImportanceParams(trip_mode=UNIFORM_LANDMARKS, landmarks_file=lm_file)
    evaluator = ImportanceEvaluator(rm, build_trip_set(rm, iparams), iparams, BlockingParams(), cache_dir=cache_dir)
    conn = Connectivity(rm, [(x, y) for x, y, _th in landmarks.values()], ctx.robot_radius)
    shares = route_shares(evaluator)
    frequent = {e for e, sh in shares.items() if sh >= float(o["route_min_trip_share"])}

    def away(x, y):
        return min(math.hypot(x - a, y - b) for a, b in keep_away) >= clearance

    jobs = []  # (type, x, y, extra)
    samples = edge_samples(rm, float(o["candidate_step_m"]))
    if "blocking" in types:
        jobs += [("blocking", x, y, None) for x, y, _e, _k, _xy in samples if away(x, y) and ctx.is_free(x, y)]
    data, w, h, res, origin = ctx.occupancy
    occ = np.asarray(data).reshape(h, w)
    if "on_route" in types:
        width = float(types["on_route"]["width_m"])
        for x, y, e, k, xy in samples:
            if e not in frequent:
                continue
            for side in (1, -1):
                spot = _against_wall(occ, origin, res, x, y, k, xy, width, side)
                if spot is not None and away(spot[0], spot[1]):
                    jobs.append(("on_route", spot[0], spot[1], (k, xy, e, spot[2])))
    if "visible" in types:
        # Distance to the nearest frequent-route cell (corridor centre lines robots often drive), metres
        on_route = np.zeros(rm.cfree.shape, dtype=bool)
        for u, v in frequent:
            poly = np.asarray(rm.graph.edges[u, v]["polyline"], dtype=int)
            on_route[poly[:, 0], poly[:, 1]] = True
        to_route = ndimage.distance_transform_edt(~on_route) * rm.res
        route_xy, route_hdg = edge_cells_xy(rm, sorted(frequent), with_heading=True)
        blocking = blocking_mask(np.asarray(data).ravel(), w, h)
        ahead = math.radians(VIEW_AHEAD_DEG)
        p = FastParams()
        from mattbot_sim.perception import CameraModel

        cam_range = CameraModel().max_range
        need = max(int(math.ceil(1.0 / rm.res)), 1)  # at least 1 m of route in view
        step = VISIBLE_STEP_M
        width = float(types["visible"]["width_m"])
        x0, y0 = origin[0] + step / 2.0, origin[1] + step / 2.0
        for gx in np.arange(x0, origin[0] + w * res, step):
            for gy in np.arange(y0, origin[1] + h * res, step):
                r, c = rm.world_to_cell(gx, gy)
                if not (0 <= r < to_route.shape[0] and 0 <= c < to_route.shape[1]):
                    continue
                if to_route[r, c] < float(o["off_route_min_m"]) or not away(gx, gy):
                    continue
                if not _free_footprint(occ, origin, res, gx, gy, width):
                    continue
                d = np.hypot(route_xy[:, 0] - gx, route_xy[:, 1] - gy)
                bearing = np.arctan2(gy - route_xy[:, 1], gx - route_xy[:, 0])
                off = np.abs((bearing - route_hdg + math.pi) % (2.0 * math.pi) - math.pi)  # 0 .. pi
                in_front = (off <= ahead) | (off >= math.pi - ahead)  # ahead in one of the two directions
                discover = (d >= p.r_min) & (d <= cam_range) & in_front
                check = (d >= p.r_min) & (d <= p.r_max)
                if discover.sum() < need or check.sum() < need:
                    continue
                either = discover | check
                clear = np.zeros(len(d), dtype=bool)
                clear[either] = width_line_of_sight_counts(blocking, origin, res, route_xy[either, 0],
                                                           route_xy[either, 1], gx, gy, width / 2.0) >= LOS_MIN_CLEAR
                n_disc, n_check = int(np.sum(clear & discover)), int(np.sum(clear & check))
                if n_disc >= need and n_check >= need:
                    jobs.append(("visible", float(gx), float(gy), (n_disc * rm.res, n_check * rm.res)))
    out = []
    for k, (typ, x, y, extra) in enumerate(jobs):
        if progress is not None:
            progress(k, len(jobs))
        width = float(types[typ]["width_m"])
        if not conn.keeps_connected(x, y, width):
            continue  # would cut a landmark off: no way around
        r = evaluator.evaluate_square("cand_%d" % k, x, y, width)
        blocked = len(r.blocked_edges) > 0
        if (typ == "blocking" and not blocked) or (typ == "on_route" and blocked):
            continue  # blocking must block an edge, on_route none (visible: limited by importance in the instance)
        cand = {"type": typ, "x": round(x, 3), "y": round(y, 3), "width": width, "I_o": round(float(r.I_o), 3),
                "frac_affected": round(float(r.frac_affected), 4), "blocked": int(blocked)}
        if typ == "on_route":
            kk, xy, e, side_m = extra
            if not _passable(ctx, x, y, width, kk, xy):
                continue
            cand["route_share"] = round(shares.get(e, 0.0), 4)
            cand["off_centre_m"] = round(side_m, 2)
        elif typ == "visible":
            cand["route_discover_m"], cand["route_in_view_m"] = round(extra[0], 2), round(extra[1], 2)
        out.append(cand)
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
    one alone leaves a way around, but two can close the same region. Obstacles are drawn type by type
    (spec.OBSTACLE_TYPES order), count of each, at least min_separation_m from every other obstacle."""
    t, o = spec.tasks, spec.obstacles
    rng = np.random.default_rng([int(seed), 20261020])
    names = list(landmarks)
    tasks, now, prev = [], float(t["start_s"]), None
    rate = spec.release_rate_per_min
    until = t.get("until_s")
    count = int(t["count"]) if until is None else None
    while (len(tasks) < count) if count is not None else True:
        now += float(rng.exponential(60.0 / rate)) if rate > 0 else 0.0
        if until is not None and now > float(until):
            break
        name = str(rng.choice([n for n in names if n != prev]))
        prev = name
        x, y, th = landmarks[name]
        tasks.append({"id": len(tasks), "release_s": round(now, 2), "landmark": name, "x": x, "y": y, "theta": th})

    chosen, usable_by_type = [], {}
    for typ, cfg in spec.obstacle_types.items():
        top = cfg.get("max_importance_m")
        usable = [c for c in candidates if c.get("type", "blocking") == typ
                  and c["I_o"] >= float(cfg["min_importance_m"]) and (top is None or c["I_o"] <= float(top))]
        usable_by_type[typ] = len(usable)
        n = 0
        for k in rng.permutation(len(usable)):
            if n >= int(cfg["count"]):
                break
            c = dict(usable[int(k)], type=typ, width=float(cfg["width_m"]), classes=list(cfg["classes"]))
            if any(math.hypot(c["x"] - d["x"], c["y"] - d["y"]) < float(o["min_separation_m"]) for d in chosen):
                continue
            if connectivity is not None and not connectivity.keeps_connected_all(
                    [(d["x"], d["y"], d["width"]) for d in chosen + [c]]):
                continue
            chosen.append(c)
            n += 1
    obstacles, events = [], []
    a0, a1 = (float(v) for v in o["appear_s"])
    l0, l1 = (float(v) for v in o["lifetime_s"])
    for i, c in enumerate(chosen):
        appear = round(float(rng.uniform(a0, a1)), 1)
        remove = round(appear + float(rng.uniform(l0, l1)), 1)
        ob = {"id": "obs_%d" % i, "type": c["type"], "class": str(rng.choice(c["classes"])), "x": c["x"], "y": c["y"],
              "width": c["width"], "I_o": c["I_o"], "appear_s": appear, "remove_s": remove}
        obstacles.append(ob)
        events.append({"at": appear, "add": {k: ob[k] for k in ("id", "class", "x", "y", "width")}})
        events.append({"at": remove, "remove": ob["id"]})
    events.sort(key=lambda e: e["at"])
    return {"spec": spec.name, "seed": int(seed), "tasks": tasks, "obstacles": obstacles, "events": events,
            "candidates_usable": sum(usable_by_type.values()), "candidates_usable_by_type": usable_by_type}


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
    blocking = [c for c in candidates if c.get("type", "blocking") == "blocking"]
    if blocking:
        sc = ax.scatter([c["x"] for c in blocking], [c["y"] for c in blocking], c=[c["I_o"] for c in blocking], s=10,
                        cmap="viridis", zorder=2, label="blocking candidates (colour = I_o)")
        fig.colorbar(sc, ax=ax, fraction=0.02, label="I_o (m per trip)")
    for typ, colour in (("on_route", "tab:orange"), ("visible", "tab:purple")):
        cs = [c for c in candidates if c.get("type") == typ]
        if cs:
            ax.scatter([c["x"] for c in cs], [c["y"] for c in cs], s=8, color=colour, zorder=2,
                       label="%s candidates" % typ)
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
        ax.annotate("%s %s I_o=%.1f\n%.0f-%.0f s" % (ob["id"], ob.get("type", "blocking"), ob["I_o"], ob["appear_s"],
                                                     ob["remove_s"]),
                    (ob["x"], ob["y"]), textcoords="offset points", xytext=(6, 6), fontsize=7, color="r")
    free = np.nonzero((occ >= 0) & (occ < 50))
    ax.set_xlim(origin[0] + free[1].min() * res - 1, origin[0] + free[1].max() * res + 1)
    ax.set_ylim(origin[1] + free[0].min() * res - 1, origin[1] + free[0].max() * res + 1)
    by_type = instance.get("candidates_usable_by_type") or {"blocking": instance["candidates_usable"]}
    ax.set_title("%s seed %d: %d tasks, %d obstacles (usable candidates: %s)" % (
        spec.name, instance["seed"], len(instance["tasks"]), len(instance["obstacles"]),
        ", ".join("%s %d" % kv for kv in by_type.items())))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
