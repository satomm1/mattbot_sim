"""Scores a run: what the ledger believes vs. what is actually in the world.

Ground truth is a list of placements (object at a spot over [t_start, t_end)), see world.Placement.
The ledger side comes from /object_beliefs (active objects with local map positions).
All times are Unix wall time, as in the ledger. Pure Python (no ROS).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

OUTCOMES = ("PRESENT", "ABSENT", "INCONCLUSIVE")


@dataclass
class LedgerObject:
    object_id: str
    class_name: str
    x: float
    y: float
    first_seen: float
    placement: Optional[dict] = None  # matched ground-truth placement when first seen


@dataclass
class RunScorer:
    match_radius: float = 1.0
    placements: List[dict] = field(default_factory=list)
    active: Dict[str, LedgerObject] = field(default_factory=dict)
    gone: Dict[str, LedgerObject] = field(default_factory=dict)  # removed from the ledger
    added: List[dict] = field(default_factory=list)
    removals: List[dict] = field(default_factory=list)
    results: List[dict] = field(default_factory=list)
    stops: Dict[str, int] = field(default_factory=lambda: {"STARTED": 0, "ENDED": 0, "ABORTED": 0})
    blockouts: List[dict] = field(default_factory=list)  # one per correct removal

    # ---------- Ground truth ----------

    def set_ground_truth(self, placements):
        self.placements = [dict(p) for p in placements]

    def placements_near(self, class_name, x, y, t=None):
        """Same-class placements within match_radius, nearest first; only those open at ``t`` if given."""
        out = []
        for p in self.placements:
            if p["class_name"] != class_name:
                continue
            if t is not None and not (p["t_start"] <= t and (p["t_end"] is None or t < p["t_end"])):
                continue
            d = math.hypot(p["x"] - x, p["y"] - y)
            if d <= self.match_radius:
                out.append((d, p))
        return [p for _d, p in sorted(out, key=lambda e: e[0])]

    def present_at(self, class_name, x, y, t):
        return bool(self.placements_near(class_name, x, y, t))

    # ---------- Ledger ----------

    def update_ledger(self, now, objects):
        """``objects``: iterable of (object_id, class_name, x, y) currently in the ledger. Returns log lines."""
        log = []
        current = {oid: (c, x, y) for oid, c, x, y in objects}
        for oid, (c, x, y) in current.items():
            if oid in self.active:
                obj = self.active[oid]
                obj.x, obj.y = x, y
                continue
            near = self.placements_near(c, x, y, now) or self.placements_near(c, x, y)
            obj = LedgerObject(oid, c, x, y, now, near[0] if near else None)
            self.active[oid] = obj
            entry = {"t": now, "object_id": oid, "class_name": c, "x": x, "y": y,
                     "true_object": obj.placement["object_id"] if obj.placement else None}
            self.added.append(entry)
            if obj.placement:
                log.append("ledger added %s (%s) = true %s" % (oid, c, obj.placement["object_id"]))
            else:
                log.append("ledger added %s (%s) at (%.2f, %.2f) with NO true object nearby (spurious)" % (oid, c, x, y))
        for oid in [o for o in self.active if o not in current]:
            obj = self.active.pop(oid)
            self.gone[oid] = obj
            log.append(self._score_removal(now, obj))
        return log

    def _score_removal(self, now, obj):
        if self.present_at(obj.class_name, obj.x, obj.y, now):
            truth = self.placements_near(obj.class_name, obj.x, obj.y, now)[0]
            self.removals.append({"t": now, "object_id": obj.object_id, "class_name": obj.class_name,
                                  "correct": False, "latency_s": None, "true_object": truth["object_id"]})
            return "FALSE REMOVAL: ledger removed %s (%s) but true %s is still there" % (
                obj.object_id, obj.class_name, truth["object_id"])
        closed = [p for p in self.placements_near(obj.class_name, obj.x, obj.y) if p["t_end"] is not None and p["t_end"] <= now]
        latest = max(closed, key=lambda p: p["t_end"]) if closed else None
        latency = now - latest["t_end"] if latest else None
        self.removals.append({"t": now, "object_id": obj.object_id, "class_name": obj.class_name, "correct": True,
                              "latency_s": latency, "true_object": latest["object_id"] if latest else None})
        self.blockouts.append({"object_id": obj.object_id, "x": obj.x, "y": obj.y, "t_removed": now, "cleared_after_s": None})
        if latency is None:
            return "ledger removed %s (%s); nothing was ever there (spurious object cleaned up)" % (obj.object_id, obj.class_name)
        return "correct removal: ledger removed %s (%s) %.1f s after true %s was taken away" % (
            obj.object_id, obj.class_name, latency, latest["object_id"])

    # ---------- Observation windows ----------

    def add_stop_event(self, event_name):
        self.stops[event_name] = self.stops.get(event_name, 0) + 1

    def add_result(self, now, object_id, class_name, outcome, reason, window_start=None, window_end=None):
        t_end = window_end or now
        t_start = window_start or t_end
        obj = self.active.get(object_id) or self.gone.get(object_id)
        truth = None
        if obj is not None:
            at_start = self.present_at(obj.class_name, obj.x, obj.y, t_start)
            at_end = self.present_at(obj.class_name, obj.x, obj.y, t_end)
            truth = at_end if at_start == at_end else "changed"
        if outcome == "INCONCLUSIVE" or truth is None:
            verdict = "n/a"
        elif truth == "changed":
            verdict = "n/a (changed during the window)"
        else:
            verdict = "correct" if (outcome == "PRESENT") == truth else "WRONG"
        self.results.append({"t": now, "object_id": object_id, "class_name": class_name, "outcome": outcome,
                             "reason": reason, "truly_present": truth, "verdict": verdict})
        return "observation of %s (%s): %s (%s), truly %s -> %s" % (
            object_id, class_name, outcome, reason,
            {True: "present", False: "gone", None: "?", "changed": "removed/added mid-window"}[truth], verdict)

    # ---------- Navigation blockouts ----------

    def update_blockouts(self, now, is_blocked):
        """``is_blocked(x, y)``: does the navigator's object layer still block this spot?"""
        log = []
        for b in self.blockouts:
            if b["cleared_after_s"] is None and not is_blocked(b["x"], b["y"]):
                b["cleared_after_s"] = now - b["t_removed"]
                log.append("navigation blockout for removed %s cleared %.1f s after the ledger removal" % (
                    b["object_id"], b["cleared_after_s"]))
        return log

    # ---------- Summary ----------

    def summary(self, now):
        removed_truths = [p for p in self.placements if p["t_end"] is not None]
        stale = []
        for obj in self.active.values():
            if not self.present_at(obj.class_name, obj.x, obj.y, now):
                closed = [p for p in self.placements_near(obj.class_name, obj.x, obj.y) if p["t_end"] is not None]
                since = now - max(p["t_end"] for p in closed) if closed else None
                stale.append({"object_id": obj.object_id, "class_name": obj.class_name, "gone_for_s": since})
        correct = [r for r in self.removals if r["correct"]]
        latencies = [r["latency_s"] for r in correct if r["latency_s"] is not None]
        outcome_counts = {o: sum(1 for r in self.results if r["outcome"] == o) for o in OUTCOMES}
        return {
            "true_objects": sorted({p["object_id"] for p in self.placements}),
            "true_removals": len(removed_truths),
            "ledger_objects_added": len(self.added),
            "spurious_ledger_objects": sum(1 for a in self.added if a["true_object"] is None),
            "correct_removals": len(correct),
            "false_removals": sum(1 for r in self.removals if not r["correct"]),
            "removal_latency_s": {"mean": sum(latencies) / len(latencies), "max": max(latencies)} if latencies else None,
            "stale_ledger_objects": stale,
            "observation_stops": dict(self.stops),
            "observation_outcomes": outcome_counts,
            "wrong_observation_outcomes": sum(1 for r in self.results if r["verdict"] == "WRONG"),
            "blockouts_still_present": [b["object_id"] for b in self.blockouts if b["cleared_after_s"] is None],
            "blockout_clear_after_s": [b["cleared_after_s"] for b in self.blockouts if b["cleared_after_s"] is not None],
        }
