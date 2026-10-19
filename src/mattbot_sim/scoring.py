"""Scores a run: what the ledger believes vs. what is actually in the world.

Ground truth is a list of placements (object at a spot over [t_start, t_end)), see world.Placement.
The ledger side comes from /object_beliefs (active objects with local map positions).
All times are ROS time (the sim clock with /use_sim_time, else the wall clock), as in the ledger. Pure Python (no ROS).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

OUTCOMES = ("PRESENT", "ABSENT", "INCONCLUSIVE")
KINDS = ("OPPORTUNISTIC", "DETOUR", "SCOUT", "AT_START", "AT_GOAL")  # mattbot_dds/ObservationStop kind constants, in order
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
    # Revalidation (localize_and_navigate._revalidate): checks skipped because they no longer passed the rules
    skips: List[str] = field(default_factory=list)  # "<object_id>: <reason>"
    detours_dropped: int = 0  # detours not started at the leave point (no target still due)
    detours_yielded: int = 0  # detours given up for another robot's earlier check (planned or under way)
    runner: Dict[str, int] = field(default_factory=lambda: {"resend": 0, "give_up": 0})
    blockouts: List[dict] = field(default_factory=list)  # one per correct removal
    # Scout decisions (observation_planner /observation/scout; decision only, nothing drives a scout yet)
    scout: Dict[str, int] = field(default_factory=lambda: {"decisions": 0, "go": 0, "go_not_idle": 0, "yielded": 0})
    scout_go_by_object: Dict[str, int] = field(default_factory=dict)  # true object id (else ledger id) -> GOs
    scout_first_go_s: Dict[str, float] = field(default_factory=dict)  # s from ledger entry to the first GO
    scout_best_net_m: Dict[str, float] = field(default_factory=dict)  # max V - C over idle decisions
    scout_last: Optional[tuple] = None  # (idle, chosen ids) of the last decision, to log changes only

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
        """A navigator log line; counts detour lifecycle steps and revalidation skips. Returns a log line
        if it was one."""
        if "Skipping check of " in text:
            rest = text.split("Skipping check of ", 1)[1].strip()
            oid, _sep, reason = rest.partition(" (")
            self.skips.append("%s: %s" % (oid, reason.rstrip(")")))
            return "navigator: " + text.strip()
        if "yielded to robot" in text and "Detour to check" in text:
            self.detours_yielded += 1
            return "navigator: " + text.strip()
        if "dropped: no longer due" in text:
            self.detours_dropped += 1
            return "navigator: " + text.strip()
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

    # ---------- Scout decisions ----------

    def _true_id(self, ledger_id):
        obj = self.active.get(ledger_id) or self.gone.get(ledger_id)
        return obj.placement["object_id"] if obj is not None and obj.placement else ledger_id

    def add_scout_decision(self, t, decision):
        """One scout decision, as a dict (both sims build the same shape):

            {"idle": bool, "go": bool, "candidates": [{"object_ids": [ledger ids], "chosen": bool, "blocked": str,
             "value_m", "cost_m", "net_m", "to_view_m", "opportunity_m", "t_away_s"}, ...]}

        Object ids are ledger ids; they are scored as the true objects they matched. Returns a log line
        when the decision changed (idle, or the chosen objects), else None.
        """
        idle, go = bool(decision.get("idle")), bool(decision.get("go"))
        cands = decision.get("candidates", [])
        chosen = sorted({self._true_id(oid) for c in cands if c.get("chosen") for oid in c["object_ids"]})
        if not idle:
            self.scout["go_not_idle"] += int(go)
        else:
            self.scout["decisions"] += 1
            self.scout["go"] += int(go)
            for c in cands:
                if "checks first" in str(c.get("blocked", "")):  # gave way to another robot's earlier check
                    self.scout["yielded"] += 1
                for oid in c["object_ids"]:
                    tid = self._true_id(oid)
                    self.scout_best_net_m[tid] = max(self.scout_best_net_m.get(tid, -math.inf), float(c["net_m"]))
                    if not c.get("chosen"):
                        continue
                    self.scout_go_by_object[tid] = self.scout_go_by_object.get(tid, 0) + 1
                    obj = self.active.get(oid) or self.gone.get(oid)
                    if tid not in self.scout_first_go_s and obj is not None:
                        self.scout_first_go_s[tid] = t - obj.first_seen
        key = (idle, tuple(chosen))
        if key == self.scout_last:
            return None
        self.scout_last = key
        if not idle:
            return "scout: not idle" + (" (but GO: %s)" % "+".join(chosen) if go else "")
        best = max(cands, key=lambda c: c["net_m"]) if cands else None
        if best is None:
            return "scout: idle, no candidates"
        return "scout: idle, %s %s: %.1f m away, V=%.1f m, C=%.1f m (opp=%.1f m, away=%.0f s)" % (
            "GO" if best.get("chosen") else "skip", "+".join(self._true_id(o) for o in best["object_ids"]),
            best.get("to_view_m", 0.0), best["value_m"], best["cost_m"], best.get("opportunity_m", 0.0),
            best.get("t_away_s", 0.0))

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
        true_ids = sorted({p["object_id"] for p in self.placements})
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
            "stops_skipped": len(self.skips),
            "stop_skip_reasons": list(self.skips),
            "detours_dropped": self.detours_dropped,
            "detours_yielded": self.detours_yielded,
            "scout_yielded": self.scout["yielded"],
            "goal_resends": self.runner["resend"],
            "goals_given_up": self.runner["give_up"],
            "wrong_observation_outcomes": sum(1 for r in self.results if r["verdict"] == "WRONG"),
            "blockouts_still_present": [b["object_id"] for b in self.blockouts if b["cleared_after_s"] is None],
            "blockout_clear_after_s": [b["cleared_after_s"] for b in self.blockouts if b["cleared_after_s"] is not None],
            "scout_decisions": self.scout["decisions"],
            "scout_go_decisions": self.scout["go"],
            "scout_go_not_idle": self.scout["go_not_idle"],
            "scout_go_by_object": dict({oid: 0 for oid in true_ids}, **self.scout_go_by_object),
            "scout_first_go_s_by_object": {k: round(v, 1) for k, v in self.scout_first_go_s.items()},
            "scout_best_net_m_by_object": {k: round(v, 2) for k, v in self.scout_best_net_m.items()},
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


# ---------- Loop rates (can the stack keep up with simulated time?) ----------


@dataclass
class RateTracker:
    """Message rate in (simulated) time, overall and per window: with sim_world's ~speed > 1, a node that
    cannot keep up publishes less often per simulated second, and the run no longer reflects the robot."""

    nominal_hz: float
    window_s: float = 10.0
    stamps: List[float] = field(default_factory=list)

    def tick(self, t):
        self.stamps.append(t)

    def summary(self):
        if len(self.stamps) < 2:
            return {"nominal_hz": self.nominal_hz, "mean_hz": None, "min_window_hz": None}
        t0, t1 = self.stamps[0], self.stamps[-1]
        mean = (len(self.stamps) - 1) / max(t1 - t0, 1e-9)
        windows = []
        k = 0
        start = t0
        while start + self.window_s <= t1:
            n = 0
            while k < len(self.stamps) and self.stamps[k] < start + self.window_s:
                if self.stamps[k] >= start:
                    n += 1
                k += 1
            windows.append(n / self.window_s)
            start += self.window_s
        return {"nominal_hz": self.nominal_hz, "mean_hz": round(mean, 2),
                "min_window_hz": round(min(windows), 2) if windows else None}

    def ok(self, fraction=0.9):
        """Mean rate within ``fraction`` of nominal (None if too few messages to tell)."""
        mean = self.summary()["mean_hz"]
        return None if mean is None else mean >= fraction * self.nominal_hz
