"""Offline detour check for a scenario: would the observation planner send the robot on a detour?

For each patrol leg (start -> waypoint 0 -> waypoint 1 -> ..., and back to waypoint 0 when the scenario
loops) the path is the roadmap shortest path (the navigator's A* path is close to it). Each object is
scored the way observation_planner.py does it, with the object's belief at 0 (fully decayed):

  - visible from the path within r_max: an opportunistic job, never a detour
  - otherwise I_o (obstacle_importance), D* and the best detour (navigation_utils/detour.py), and
    V - C from ThresholdPolicy.detour_options; "GO" if the policy would take it (never for a detour
    that branches off within skip_start_m / skip_goal_m of the leg's start or goal)

It also reports whether the sim camera (5 m, any heading, line of sight on the sim map) could see the
object from the path, which would let the mapper find the object by itself while patrolling.

Needs mattbot_navigation's navigation_utils (and path_planning's social_path_planning) on the path;
no ROS. Used by scripts/check_detour_scenario.py and test/test_detour_scenarios.py.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mattbot_sim.world import World, load_maps


@dataclass
class LegResult:
    leg: str
    visible_from_path: bool
    d_star: Optional[float] = None
    detour_m: Optional[float] = None
    value_m: Optional[float] = None
    cost_m: Optional[float] = None
    viewpoint: Optional[tuple] = None
    branch_arc_m: Optional[float] = None  # where the detour leaves the path (arc length)
    blocked: str = ""  # why it is never taken (branches off at the start / goal)
    go: bool = False

    @property
    def net_m(self):
        return None if self.value_m is None else self.value_m - self.cost_m


@dataclass
class ObjectResult:
    object_id: str
    importance_m: float
    camera_sees_from_path: bool
    legs: List[LegResult] = field(default_factory=list)

    @property
    def hidden(self):
        return not any(leg.visible_from_path for leg in self.legs)

    @property
    def detour_go(self):
        return any(leg.go for leg in self.legs)

    def best_leg(self):
        scored = [leg for leg in self.legs if leg.net_m is not None]
        return max(scored, key=lambda leg: leg.net_m) if scored else None


def occupancy_tuple(grid):
    """GridMap -> (data, width, height, resolution, origin) as the planner gets it from /map (int8 data,
    so the roadmap cache fingerprint matches the one observation_planner builds)."""
    data = np.asarray(grid.occupancy, dtype=np.int8).ravel()
    return data, grid.width, grid.height, grid.resolution, (grid.origin_x, grid.origin_y)


def patrol_legs(scenario):
    stops = [(scenario.start.x, scenario.start.y)] + [(w.x, w.y) for w in scenario.waypoints]
    legs = list(zip(stops[:-1], stops[1:]))
    if scenario.loop and len(scenario.waypoints) > 1:
        legs.append((stops[-1], stops[1]))
    return legs


def camera_sees(world, path_xy, obj, max_range=5.0, step_m=0.25):
    """Could the sim camera see ``obj`` from some point of the path (any heading)?"""
    last = None
    for x, y in path_xy:
        if last is not None and math.hypot(x - last[0], y - last[1]) < step_m:
            continue
        last = (x, y)
        if math.hypot(obj.x - x, obj.y - y) <= max_range and world.line_of_sight(x, y, obj):
            return True
    return False


def check_scenario(scenario, map_json_dir, cache_dir, n_trips=5.0, margin_m=0.0, max_detour_m=15.0,
                   r_min=1.0, r_max=3.5, robot_radius=0.4, cruise_speed=0.4, dwell_s=4.0, skip_start_m=1.0,
                   skip_goal_m=1.0, object_ids=None):
    """ObjectResult per scenario object (or per ``object_ids``)."""
    from navigation_utils.detour import DetourParams, DetourPlanner
    from navigation_utils.obstacle_importance import ImportanceEvaluator, ImportanceParams, build_trip_set
    from navigation_utils.observation_policy import Candidate, DetourValueParams, ThresholdPolicy
    from navigation_utils.roadmap import RoadmapParams, load_or_build_roadmap
    from navigation_utils.viewpoints import blocking_mask, compute_viewshed, nearby_blockers

    grid, _grid_mod = load_maps(scenario.map, map_json_dir)  # the planner sees /map, not /map_mod
    occupancy = occupancy_tuple(grid)
    data, width, height, res, origin = occupancy
    roadmap, _cached = load_or_build_roadmap(*occupancy, params=RoadmapParams(robot_radius=robot_radius),
                                             cache_dir=cache_dir)
    planner = DetourPlanner(roadmap)
    iparams = ImportanceParams()  # observation_planner defaults: 50 random landmarks, uniform trips
    evaluator = ImportanceEvaluator(roadmap, build_trip_set(roadmap, iparams), iparams, cache_dir=cache_dir)
    policy = ThresholdPolicy(cruise_speed=cruise_speed, dwell_s=dwell_s,
                             detour_params=DetourValueParams(n_trips=n_trips, margin_m=margin_m,
                                                             hard_cap_m=max_detour_m, skip_start_m=skip_start_m,
                                                             skip_goal_m=skip_goal_m))
    dparams = DetourParams(max_detour_m=max_detour_m, r_max=r_max)

    objects = [o for o in scenario.objects if object_ids is None or o.object_id in object_ids]
    all_objects = [(o.object_id, o.x, o.y, o.width) for o in scenario.objects]
    blocking = blocking_mask(data, width, height)
    world = World(grid, scenario.objects)
    importance = {o.object_id: evaluator.evaluate_square(o.object_id, o.x, o.y, o.width).I_o for o in objects}
    policy.importance_fn = policy.importance_m_fn = lambda c: importance[c.object_id]

    views = {}
    for o in objects:
        blockers, _sig = nearby_blockers(o.object_id, (o.x, o.y), all_objects, r_max + 1.0)
        views[o.object_id] = compute_viewshed(blocking, (o.x, o.y), origin, res, r_min, r_max, blockers=blockers)

    paths = []
    for a, b in patrol_legs(scenario):
        path = planner.shortest_path_xy(a, b)
        if path is not None:
            paths.append(("(%.1f, %.1f) -> (%.1f, %.1f)" % (a + b), np.asarray(path)))

    results = []
    for o in objects:
        c = Candidate(o.object_id, o.class_name, o.x, o.y, 0.0, o.width)
        vs = views[o.object_id]
        r = ObjectResult(o.object_id, importance[o.object_id],
                         any(camera_sees(world, p, o) for _name, p in paths))
        for name, path in paths:
            leg = LegResult(name, policy.checkable_from_path(path, vs))
            if not leg.visible_from_path:
                leg.d_star = policy.max_detour_m(c)
                if leg.d_star > 0:
                    detours = planner.compute(path, {o.object_id: (o.x, o.y, planner.viewpoint_nodes(vs), leg.d_star)},
                                              dparams)
                    chosen, evaluations = policy.detour_options(path, [c], {o.object_id: vs}, 0.0, detours=detours)
                    if evaluations:
                        ev = evaluations[0]
                        leg.detour_m, leg.value_m, leg.cost_m = ev.detour_m, ev.value_m, ev.cost_m
                        leg.viewpoint = (round(ev.option.stop.x, 2), round(ev.option.stop.y, 2))
                        leg.branch_arc_m = detours[o.object_id].branch_arc_m
                        leg.blocked = ev.blocked
                        leg.go = bool(ev.chosen)
            r.legs.append(leg)
        results.append(r)
    return results


def format_results(results):
    lines = ["%-12s %6s %7s %6s  %-28s %6s %6s %6s %6s  %s" % (
        "object", "I_o", "camera", "D*", "leg", "D", "V", "C", "V-C", "decision")]
    for r in results:
        for k, leg in enumerate(r.legs):
            head = ("%-12s %6.2f %7s" % (r.object_id, r.importance_m, "sees" if r.camera_sees_from_path else "-")
                    if k == 0 else " " * 27)
            if leg.visible_from_path:
                decision = "seen from path (opportunistic)"
            elif leg.d_star is not None and leg.d_star <= 0:
                decision = "never worth a detour (D* <= 0)"
            elif leg.value_m is None:
                decision = "no viewpoint within D*"
            elif leg.go:
                decision = "GO -> viewpoint (%.2f, %.2f), leaves at %.1f m" % (leg.viewpoint + (leg.branch_arc_m,))
            elif leg.blocked:
                decision = "skip: %s (leaves at %.1f m)" % (leg.blocked, leg.branch_arc_m)
            else:
                decision = "skip (V - C <= margin)"
            fmt = lambda v: "-" if v is None else "%.1f" % v  # noqa: E731
            lines.append("%s %6s  %-28s %6s %6s %6s %6s  %s" % (
                head, fmt(leg.d_star), leg.leg, fmt(leg.detour_m), fmt(leg.value_m), fmt(leg.cost_m),
                fmt(leg.net_m), decision))
    return "\n".join(lines)
