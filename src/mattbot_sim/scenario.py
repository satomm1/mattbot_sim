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

Fleet scenarios (several simulated robots, one ROS master each; scripts/run_fleet.py) give a
``robots`` list instead of ``start`` / ``waypoints``; objects and events are shared:

    fleet_start_s: 60             # patrols and event times start this long after the shared clock epoch
    robots:
      - id: 91                    # ROBOT_ID (DDS id); avoid ids of real robots
        start: {x: 28.0, y: 18.95, theta: 0.0}
        waypoints: [...]
        loop: true                # optional, else the top-level value
        expect: {...}             # this robot's checks, merged over the top-level expect
    launch_args: {observe_detour: true}   # optional: sim.launch args run_fleet.py gives every robot

A waypoint may have ``pause_s`` (wait there this long before the next one; default ~pause_at_waypoint_s).

load_scenario(path, robot_id) picks that robot's part (the first robot if robot_id is None).

The same event dicts (without ``at``) can be sent at runtime as YAML on /sim/event.
Pure Python (no ROS).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    robot_id: Optional[int] = None  # fleet scenarios: the robot this view is for
    robot_ids: List[int] = field(default_factory=list)  # fleet scenarios: every robot, in file order
    fleet_start_s: Optional[float] = None  # fleet scenarios: patrol / event time 0 = clock epoch + this
    waypoint_pauses: List[Optional[float]] = field(default_factory=list)  # per waypoint pause_s (None: default)
    launch_args: Dict[str, str] = field(default_factory=dict)  # sim.launch args run_fleet.py passes every robot

    @property
    def is_fleet(self):
        return bool(self.robot_ids)


def parse_expect(expect, path):
    expect = expect or {}
    if not isinstance(expect, dict):
        raise ValueError("%s: expect must be a mapping" % path)
    return {str(k): float(v) for k, v in expect.items()}


def select_robot(d, path, robot_id):
    """Fleet scenario dict -> (robot dict, all ids); robot_id None picks the first robot."""
    robots = d["robots"]
    if not isinstance(robots, list) or not robots:
        raise ValueError("%s: robots must be a non-empty list" % path)
    for r in robots:
        if "id" not in r or "start" not in r:
            raise ValueError("%s: every robot needs id and start, got %r" % (path, r))
    ids = [int(r["id"]) for r in robots]
    if len(set(ids)) != len(ids):
        raise ValueError("%s: duplicate robot ids" % path)
    if robot_id is None:
        return robots[0], ids
    for r in robots:
        if int(r["id"]) == int(robot_id):
            return r, ids
    raise ValueError("%s: no robot with id %s (robots: %s)" % (path, robot_id, ids))


def robot_id_from_env():
    """ROBOT_ID of this process (sim.launch sets it for every node), or None."""
    value = os.environ.get("ROBOT_ID", "")
    return int(value) if value.strip().lstrip("-").isdigit() else None


def load_scenario(path, robot_id=None):
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    robot, robot_ids = (None, [])
    if "robots" in d:
        robot, robot_ids = select_robot(d, path, robot_id)
        for key in ("start", "waypoints"):
            if key in d:
                raise ValueError("%s: fleet scenarios give %r per robot" % (path, key))
    required = ("map",) if robot else ("map", "start")
    for key in required:
        if key not in d:
            raise ValueError("%s: missing %r" % (path, key))
    part = robot or d  # where start / waypoints / loop come from
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
    expect = parse_expect(d.get("expect"), path)
    if robot:
        expect.update(parse_expect(robot.get("expect"), path))
    return Scenario(
        name=str(d.get("name", os.path.splitext(os.path.basename(path))[0])),
        map=str(d["map"]),
        start=parse_pose(part["start"], "start"),
        objects=objects,
        waypoints=[parse_pose(w, "waypoint") for w in part.get("waypoints", [])],
        loop=bool(part.get("loop", d.get("loop", True))),
        duration=float(d.get("duration", 0.0)),
        events=events,
        peer_id=int(d.get("peer_id", DEFAULT_PEER_ID)),
        known_from_peer=known_from_peer,
        detour=detour,
        expect=expect,
        robot_id=int(robot["id"]) if robot else None,
        robot_ids=robot_ids,
        fleet_start_s=float(d.get("fleet_start_s", 60.0)) if robot else None,
        waypoint_pauses=[float(w["pause_s"]) if isinstance(w, dict) and "pause_s" in w else None
                         for w in part.get("waypoints", [])],
        launch_args={str(k): str(v).lower() if isinstance(v, bool) else str(v)
                     for k, v in (d.get("launch_args") or {}).items()},
    )
