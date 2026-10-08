"""The observation planner and evaluator nodes, as library calls.

ObservationCore (shared by the fleet): viewsheds on the static /map, the roadmap / detour planner and the
importance I_o of every ledger object, all robot-independent. CheckPlanner (per robot): its ThresholdPolicy
(per-robot cooldowns) and select(), following observation_planner.select_srv. Evaluator (per robot):
observation windows, following observation_evaluator.py (use_depth off): frame_verdict per detector frame,
window_outcome at the end, then PRESENT -> confirm, ABSENT -> remove, and re-sightings of other objects.
"""

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from mattbot_sim.fast import imports  # noqa: F401
from navigation_utils.observation_policy import DETOUR, Candidate, DetourValueParams, ThresholdPolicy
from navigation_utils.viewpoints import blocking_mask, compute_viewshed, nearby_blockers
from observation_eval.evidence import (
    OUTCOME_ABSENT, OUTCOME_PRESENT, CameraPose, Detection, EvalParams, Intrinsics, Target, frame_verdict,
    match_known, window_outcome,
)

RECOMPUTE_MOVE_M = 0.2  # observation_planner: recompute a viewshed if the object moved further
OUTCOMES = ("PRESENT", "ABSENT", "INCONCLUSIVE")


@dataclass
class StopRequest:
    """One ObservationStop (mattbot_dds/msg/ObservationStop) as the navigator gets it."""

    kind: int
    resume: int
    path_index: int
    x: float
    y: float
    heading: float
    object_ids: List[str]
    targets: List[tuple]
    cost_s: float


class ObservationCore:
    """Fleet-wide, robot-independent parts of the observation planner."""

    def __init__(self, grid, params, cache_dir=None):
        self.params = params
        self.grid = grid
        self.origin = (grid.origin_x, grid.origin_y)
        self.blocking = blocking_mask(np.asarray(grid.occupancy).ravel(), grid.width, grid.height)
        self.viewsheds = {}  # object_id -> (Viewshed, signature)
        self.detour_planner = None
        self.tracker = None
        if params.observe_detour:
            self._build_importance(cache_dir)

    def _build_importance(self, cache_dir):
        from navigation_utils.detour import DetourPlanner
        from navigation_utils.obstacle_importance import (
            ImportanceEvaluator, ImportanceParams, ObjectImportanceTracker, build_trip_set,
        )
        from navigation_utils.roadmap import RoadmapParams, load_or_build_roadmap

        from mattbot_sim.detour_check import occupancy_tuple

        cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "mattbot_fastsim_roadmap")
        roadmap, _cached = load_or_build_roadmap(*occupancy_tuple(self.grid), params=RoadmapParams(robot_radius=0.4),
                                                 cache_dir=cache_dir)
        self.detour_planner = DetourPlanner(roadmap)
        iparams = ImportanceParams()  # observation_planner defaults
        self.tracker = ObjectImportanceTracker(ImportanceEvaluator(roadmap, build_trip_set(roadmap, iparams), iparams))

    def update_objects(self, beliefs):
        """Importance of every ledger object (synchronous; the robot computes it in a background worker)."""
        if self.tracker is not None:
            self.tracker.update({b.object_id: (b.x, b.y, b.width) for b in beliefs})

    def importance_m(self, c):
        result = self.tracker.get(c.object_id) if self.tracker is not None else None
        return None if result is None else result.I_o

    def importance(self, c):
        v = self.importance_m(c)
        return 1.0 if v is None else v

    def ensure_viewsheds(self, wanted, candidates):
        p = self.params
        all_objects = [(o.object_id, o.x, o.y, o.width) for o in candidates]
        for c in wanted:
            blockers, signature = nearby_blockers(c.object_id, (c.x, c.y), all_objects, p.r_max + 1.0)
            cached = self.viewsheds.get(c.object_id)
            if cached is not None:
                vs, sig = cached
                if sig == signature and math.hypot(vs.obj_xy[0] - c.x, vs.obj_xy[1] - c.y) <= RECOMPUTE_MOVE_M:
                    continue
            vs = compute_viewshed(self.blocking, (c.x, c.y), self.origin, self.grid.resolution, p.r_min, p.r_max,
                                  blockers=blockers)
            self.viewsheds[c.object_id] = (vs, signature)
        live = {c.object_id for c in candidates}
        for oid in [oid for oid in self.viewsheds if oid not in live]:
            del self.viewsheds[oid]


class CheckPlanner:
    """One robot's observation planner: ThresholdPolicy with its own cooldowns (record_check on ENDED)."""

    def __init__(self, core):
        self.core = core
        p = core.params
        self.policy = ThresholdPolicy(
            check_below_belief=p.observe_check_below_belief, cooldown_s=p.observe_cooldown_s, max_cost_s=0.0,
            r_pref=2.0, skip_start_m=0.5, skip_goal_m=0.8, merge_m=0.5, max_turn=math.pi / 2, turn_rate=1.0,
            dwell_s=p.observe_dwell_s, cruise_speed=p.cruising_velocity,
        )
        if core.tracker is not None:
            self.policy.importance_fn = core.importance
            self.policy.importance_m_fn = core.importance_m
            self.policy.detour_params = DetourValueParams(
                n_trips=p.observe_detour_n_trips, conclusive_prob=1.0, margin_m=p.observe_detour_margin_m,
                max_detours_per_path=1, hard_cap_m=p.observe_max_detour_m)
        self.detour_log = []  # (object ids, detour_m, V, C, chosen, blocked) of the last request

    def record_check(self, object_id, t):
        self.policy.record_check(object_id, t)

    def select(self, path_xy, beliefs, now, exclude=()):
        """observation_planner.select_srv: path (strided trajectory points) -> [StopRequest]."""
        path_xy = np.asarray(path_xy, dtype=float)
        if len(path_xy) < 2:
            return []
        core, pol = self.core, self.policy
        cands = [Candidate(b.object_id, b.class_name, b.x, b.y, b.belief, b.width) for b in beliefs]
        exclude = set(exclude)
        eligible = pol.eligible_candidates(cands, now, exclude)
        detour_eligible = [c for c in cands if pol.detour_eligible(c, now, exclude)] if core.tracker else []
        core.ensure_viewsheds({c.object_id: c for c in eligible + detour_eligible}.values(), cands)
        views = {oid: vs for oid, (vs, _sig) in core.viewsheds.items()}
        options = pol.opportunistic_options(path_xy, cands, views, now, exclude)
        self.detour_log = []
        if core.tracker is not None and core.detour_planner is not None:
            from navigation_utils.detour import DetourParams

            dcands = pol.detour_candidates(path_xy, cands, views, now, exclude)
            targets = {}
            for c in dcands:
                d_star = pol.max_detour_m(c)
                if d_star > 0:
                    targets[c.object_id] = (c.x, c.y, core.detour_planner.viewpoint_nodes(views[c.object_id]), d_star)
            detours = core.detour_planner.compute(
                path_xy, targets, DetourParams(max_detour_m=core.params.observe_max_detour_m, r_max=core.params.r_max))
            chosen, evaluations = pol.detour_options(path_xy, cands, views, now, exclude, detours)
            options += chosen
            self.detour_log = [(ev.object_ids, ev.detour_m, ev.value_m, ev.cost_m, ev.chosen, ev.blocked)
                               for ev in evaluations]
        return [StopRequest(o.kind, o.resume, o.path_index, o.stop.x, o.stop.y, o.stop.heading,
                            [c.object_id for c in o.candidates], [(c.x, c.y) for c in o.candidates], o.cost_s)
                for o in options]


@dataclass
class Window:
    object_id: str
    target: Target
    start: float
    params: EvalParams
    frames: list = field(default_factory=list)
    others: Dict[str, list] = field(default_factory=dict)  # object_id -> [matches, frame count]


class Evaluator:
    """One robot's observation evaluator (use_depth off)."""

    def __init__(self, params, camera):
        self.params = EvalParams(use_depth=False, min_absent_frames=int(params.observe_min_absent_frames))
        self.intrinsics = Intrinsics(camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height)
        self.resight_min_interval_s = params.resight_min_interval_s
        self.windows: Dict[str, Window] = {}
        self.last_resight: Dict[str, float] = {}

    def start(self, object_id, target_xy, now, beliefs):
        known = {b.object_id: b for b in beliefs}.get(object_id)
        class_name, width = (known.class_name, known.width) if known else ("", 0.5)
        params = self.params
        if not class_name:
            params = EvalParams(**{**params.__dict__, "match_any_class": True})
        self.windows[object_id] = Window(object_id, Target(class_name, target_xy[0], target_xy[1], width), now, params)

    def abort(self, object_id):
        self.windows.pop(object_id, None)

    def frame(self, detections, cam_pose, beliefs):
        """One detector frame (detections: mattbot_sim.perception.Detection, map frame)."""
        if not self.windows:
            return
        dets = [Detection(d.class_name, d.x, d.y, d.width) for d in detections]
        cam = CameraPose(*cam_pose)
        objects = {b.object_id: (b.class_name, b.x, b.y, b.width) for b in beliefs}
        for target_id, w in self.windows.items():
            others = {oid: o for oid, o in objects.items() if oid != target_id}
            w.frames.append(frame_verdict(dets, w.target, cam, self.intrinsics, w.params, localization_ok=True,
                                          known_blockers=[(x, y, wd) for _c, x, y, wd in others.values()]))
            for oid, matches in match_known(dets, others, w.params).items():
                entry = w.others.setdefault(oid, [[], 0])
                entry[0].extend(matches)
                entry[1] += 1

    def end(self, object_id, now, beliefs):
        """ENDED: returns (result dict, actions) with actions [("confirm"|"remove", class, x, y, width)]."""
        w = self.windows.pop(object_id, None)
        if w is None:
            return None, []
        out = window_outcome(w.frames, w.params)
        actions = []
        if out.outcome == OUTCOME_PRESENT:
            self.last_resight[object_id] = now
            matches = [m for f in w.frames for m in f.matches]
            actions.append(("confirm", w.target.class_name, float(np.mean([m[0] for m in matches])),
                            float(np.mean([m[1] for m in matches])), float(np.mean([m[2] for m in matches]))))
        known = {b.object_id: b for b in beliefs}
        resighted = []
        for oid, (matches, n_frames) in sorted(w.others.items()):
            if n_frames < w.params.min_present_frames or now - self.last_resight.get(oid, -math.inf) < \
                    self.resight_min_interval_s or oid not in known:
                continue
            self.last_resight[oid] = now
            actions.append(("confirm", known[oid].class_name, float(np.mean([m[0] for m in matches])),
                            float(np.mean([m[1] for m in matches])), float(np.mean([m[2] for m in matches]))))
            resighted.append(oid)
        if out.outcome == OUTCOME_ABSENT and w.target.class_name:
            actions.append(("remove", w.target.class_name, w.target.x, w.target.y, w.target.width))
        result = {"object_id": object_id, "class_name": w.target.class_name, "outcome": OUTCOMES[out.outcome],
                  "reason": out.reason, "window_start": w.start, "window_end": now, "frames": out.frames,
                  "resighted": resighted}
        return result, actions


def is_detour(stop):
    return stop.kind == DETOUR
