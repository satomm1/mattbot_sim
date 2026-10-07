"""Scenario files (scenarios/*.yaml): map, start pose, ground-truth objects, patrol waypoints, timed events.

    name: remove_one
    map: current_map              # mattbot_mcl/map_json stem, or a path to a map_server .yaml
    start: {x: 28.0, y: 18.95, theta: 0.0}
    objects:
      - {id: chair_1, class: chair, x: 35.0, y: 19.6, width: 0.5}
    waypoints:
      - {x: 46.0, y: 18.95, theta: 3.14}
      - {x: 28.0, y: 18.95, theta: 0.0}
    loop: true                    # repeat the waypoints
    duration: 600                 # s after the robot is localized; 0 = run until Ctrl-C
    events:                       # times in s after the robot is localized
      - {at: 120, remove: chair_1}
      - {at: 300, restore: chair_1}
      - {at: 400, move: chair_1, x: 36.5, y: 19.6}
      - {at: 450, add: {id: cone_2, class: cone, x: 40.0, y: 18.3, width: 0.3}}

Optional, for detour tests:

    peer_id: 98                   # observer id of the simulated peer robot (default 98)
    objects:
      - {id: cone_9, class: cone, x: 25.5, y: 25.0, width: 0.4,
         known_from_peer: true,   # put it in the ledger at start, as if peer_id had seen it
         detour: expected}        # offline check (check_detour_scenario.py): expected | declined
    expect:                       # checked by sim_monitor at the end of the run (scoring.check_expectations)
      min_detour_stops: 1         # min_<field> / max_<field> / <field> (==) on summary fields
      false_removals: 0

The same event dicts (without ``at``) can be sent at runtime as YAML on /sim/event.
Pure Python (no ROS).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

import yaml

from mattbot_sim.kinematics import Pose2D
from mattbot_sim.world import SimObject

ACTIONS = ("remove", "restore", "move", "add")
DETOUR_ANNOTATIONS = ("expected", "declined")
DEFAULT_PEER_ID = 98


def parse_object(d):
    for key in ("id", "class", "x", "y"):
        if key not in d:
            raise ValueError("object %r is missing %r" % (d, key))
    return SimObject(str(d["id"]), str(d["class"]), float(d["x"]), float(d["y"]),
                     float(d.get("width", 0.5)), bool(d.get("present", True)))


def parse_pose(d, what):
    if not isinstance(d, dict) or "x" not in d or "y" not in d:
        raise ValueError("%s must be {x: .., y: .., theta: ..}, got %r" % (what, d))
    return Pose2D(float(d["x"]), float(d["y"]), float(d.get("theta", 0.0)))


def parse_event(d):
    """Validate one world event; returns a normalised dict (``add`` holds a SimObject)."""
    if not isinstance(d, dict):
        raise ValueError("event must be a mapping, got %r" % (d,))
    actions = [a for a in ACTIONS if a in d]
    if len(actions) != 1:
        raise ValueError("event %r must have exactly one of %s" % (d, ", ".join(ACTIONS)))
    action = actions[0]
    out = {}
    if "at" in d:
        out["at"] = float(d["at"])
    if action == "add":
        out["add"] = parse_object(d["add"])
    else:
        out[action] = str(d[action])
    if action == "move":
        if "x" not in d or "y" not in d:
            raise ValueError("move event %r needs x and y" % (d,))
        out["x"], out["y"] = float(d["x"]), float(d["y"])
    return out


@dataclass
class Scenario:
    name: str
    map: str
    start: Pose2D
    objects: List[SimObject]
    waypoints: List[Pose2D]
    loop: bool = True
    duration: float = 0.0
    events: List[dict] = field(default_factory=list)
    peer_id: int = DEFAULT_PEER_ID
    known_from_peer: List[str] = field(default_factory=list)  # object ids seeded into the ledger at start
    detour: Dict[str, str] = field(default_factory=dict)  # object_id -> "expected" | "declined"
    expect: Dict[str, float] = field(default_factory=dict)  # see scoring.check_expectations


def load_scenario(path):
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    for key in ("map", "start"):
        if key not in d:
            raise ValueError("%s: missing %r" % (path, key))
    objects = [parse_object(o) for o in d.get("objects", [])]
    ids = [o.object_id for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("%s: duplicate object ids" % path)
    events = sorted((parse_event(e) for e in d.get("events", [])), key=lambda e: e.get("at", 0.0))
    for e in events:
        if "at" not in e:
            raise ValueError("%s: scenario event %r needs 'at'" % (path, e))
    raw = d.get("objects", [])
    known_from_peer = [str(o["id"]) for o in raw if o.get("known_from_peer", False)]
    detour = {}
    for o in raw:
        if "detour" in o:
            if o["detour"] not in DETOUR_ANNOTATIONS:
                raise ValueError("%s: object %s: detour must be one of %s" % (path, o["id"], ", ".join(DETOUR_ANNOTATIONS)))
            detour[str(o["id"])] = str(o["detour"])
    expect = d.get("expect", {}) or {}
    if not isinstance(expect, dict):
        raise ValueError("%s: expect must be a mapping" % path)
    return Scenario(
        name=str(d.get("name", os.path.splitext(os.path.basename(path))[0])),
        map=str(d["map"]),
        start=parse_pose(d["start"], "start"),
        objects=objects,
        waypoints=[parse_pose(w, "waypoint") for w in d.get("waypoints", [])],
        loop=bool(d.get("loop", True)),
        duration=float(d.get("duration", 0.0)),
        events=events,
        peer_id=int(d.get("peer_id", DEFAULT_PEER_ID)),
        known_from_peer=known_from_peer,
        detour=detour,
        expect={str(k): float(v) for k, v in expect.items()},
    )
