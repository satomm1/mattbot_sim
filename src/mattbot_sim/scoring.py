"""Scores a run: what the ledger believes vs. what is actually in the world.

Ground truth is a list of placements (object at a spot over [t_start, t_end)), see world.Placement.
The ledger side comes from /object_beliefs (active objects with local map positions).
All times are Unix wall time, as in the ledger. Pure Python (no ROS).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

OUTCOMES = ("PRESENT", "ABSENT", "INCONCLUSIVE")
KINDS = ("OPPORTUNISTIC", "DETOUR")  # mattbot_dds/ObservationStop kind constants, in order
STOP_EVENTS = ("STARTED", "ENDED", "ABORTED")

# Navigator log lines (localize_and_navigate.py _start_detour / _arrive_at_detour_viewpoint / detour resume /
# _abandon_detour / _cancel_detour). Abandons and cancels publish no ObservationEvent, so the log is the
# only way to see them. Keep in sync with the navigator.
DETOUR_LOG = (
    ("started", "Detour to ("),
    ("reached", "Reached detour viewpoint"),
    ("done", "Detour done; replanning to the goal"),
    ("abandoned", "abandoned ("),
    ("cancelled", "Detour cancelled ("),
)


def kind_name(kind):
    return KINDS[kind] if isinstance(kind, int) and 0 <= kind < len(KINDS) else str(kind)


def _paren(text, after):
    """Text inside the parentheses that follow ``after`` (the reason of an abandon / cancel)."""
    i = text.find(after)
    if i < 0:
        return ""
    rest = text[i + len(after):]
    return rest[:rest.find(")")] if ")" in rest else rest


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
    stops: Dict[str, int] = field(default_factory=lambda: {e: 0 for e in STOP_EVENTS})
    stops_by_kind: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {k: {e: 0 for e in STOP_EVENTS} for k in KINDS})
    stop_log: List[dict] = field(default_factory=list)  # every ObservationEvent with kind and robot pose
    last_kind: Dict[str, str] = field(default_factory=dict)  # object_id -> kind of its latest stop
    detours: Dict[str, int] = field(default_factory=lambda: {name: 0 for name, _ in DETOUR_LOG})
    detour_reasons: List[str] = field(default_factory=list)  # abandon / cancel reasons
    runner: Dict[str, int] = field(default_factory=lambda: {"resend": 0, "give_up": 0})
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

    def add_stop_event(self, event_name, kind="OPPORTUNISTIC", object_id=None, t=None, robot_xy=None, distance=None):
        """One ObservationEvent. Returns a log line for STARTED / ABORTED, else None."""
        self.stops[event_name] = self.stops.get(event_name, 0) + 1
        by_event = self.stops_by_kind.setdefault(kind, {e: 0 for e in STOP_EVENTS})
        by_event[event_name] = by_event.get(event_name, 0) + 1
        if object_id is not None:
            self.last_kind[object_id] = kind
        self.stop_log.append({"t": t, "event": event_name, "kind": kind, "object_id": object_id,
                              "robot_xy": list(robot_xy) if robot_xy else None, "distance": distance})
        where = " from (%.2f, %.2f)" % tuple(robot_xy) if robot_xy else ""
        if event_name == "STARTED":
            return "%s stop STARTED for %s at %.1f m%s" % (kind, object_id, distance or 0.0, where)
        if event_name == "ABORTED":
            return "%s stop ABORTED for %s%s" % (kind, object_id, where)
        return None

    def add_nav_log(self, text):
        """A navigator log line; counts detour lifecycle steps. Returns a log line if it was one."""
        for name, pattern in DETOUR_LOG:
            if pattern in text and (name != "abandoned" or "Detour to check" in text):
                self.detours[name] += 1
                if name in ("abandoned", "cancelled"):
                    self.detour_reasons.append("%s: %s" % (name, _paren(text, pattern)))
                return "navigator: " + text.strip()
        return None

    def add_runner_event(self, text):
        """A /sim/runner_event line ("resend goal ...", "give_up goal ...")."""
        key = text.split(" ", 1)[0]
        if key in self.runner:
            self.runner[key] += 1
        return "scenario_runner: " + text

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
        kind = self.last_kind.get(object_id)  # results follow the stop's ENDED event
        self.results.append({"t": now, "object_id": object_id, "class_name": class_name, "outcome": outcome,
                             "reason": reason, "truly_present": truth, "verdict": verdict, "kind": kind})
        return "%sobservation of %s (%s): %s (%s), truly %s -> %s" % (
            kind + " " if kind else "", object_id, class_name, outcome, reason,
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
        outcomes_by_kind = {k: {o: sum(1 for r in self.results if r["outcome"] == o and r["kind"] == k) for o in OUTCOMES}
                            for k in KINDS}
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
            "observation_stops_by_kind": {k: dict(v) for k, v in self.stops_by_kind.items()},
            "observation_outcomes_by_kind": outcomes_by_kind,
            "detour_stops": self.stops_by_kind.get("DETOUR", {}).get("ENDED", 0),
            "detours_started": self.detours["started"],
            "detours_reached": self.detours["reached"],
            "detours_done": self.detours["done"],
            "detours_abandoned": self.detours["abandoned"],
            "detours_cancelled": self.detours["cancelled"],
            "detour_abandon_reasons": list(self.detour_reasons),
            "goal_resends": self.runner["resend"],
            "goals_given_up": self.runner["give_up"],
            "wrong_observation_outcomes": sum(1 for r in self.results if r["verdict"] == "WRONG"),
            "blockouts_still_present": [b["object_id"] for b in self.blockouts if b["cleared_after_s"] is None],
            "blockout_clear_after_s": [b["cleared_after_s"] for b in self.blockouts if b["cleared_after_s"] is not None],
        }


# ---------- Expectations ----------


def summary_value(summary, path):
    """Summary field by dotted path (e.g. observation_outcomes_by_kind.DETOUR.PRESENT); lists count their items."""
    value = summary
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    if isinstance(value, (list, tuple)):
        return len(value)
    return value


def check_expectations(summary, expect):
    """Scenario ``expect:`` block against a summary. Keys are summary fields (dotted paths allowed) with an
    optional min_ / max_ prefix; no prefix means equal. Returns one dict per key with ``ok``."""
    out = []
    for key, expected in expect.items():
        op, path = "==", key
        for prefix, o in (("min_", ">="), ("max_", "<=")):
            if key.startswith(prefix):
                op, path = o, key[len(prefix):]
        try:
            actual = summary_value(summary, path)
        except KeyError:
            out.append({"key": key, "op": op, "expected": expected, "actual": None, "ok": False,
                        "error": "no summary field %r" % path})
            continue
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            ok = False
        elif op == ">=":
            ok = actual >= expected
        elif op == "<=":
            ok = actual <= expected
        else:
            ok = actual == expected
        out.append({"key": key, "op": op, "expected": expected, "actual": actual, "ok": bool(ok)})
    return out
