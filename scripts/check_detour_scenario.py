#!/usr/bin/env python3
"""Offline check of a scenario's detour targets (no ROS master needed; see mattbot_sim/detour_check.py).

  rosrun mattbot_sim check_detour_scenario.py $(rospack find mattbot_sim)/scenarios/detour_present.yaml \
      [--n-trips 5] [--margin 0] [--max-detour 15] [--cache-dir ~/.ros/mattbot_roadmap]

Prints, per object and patrol leg, whether the path sees it, its importance I_o, and the detour the
observation planner would choose with the object's belief fully decayed. Exits 1 if an object annotated
``detour: expected`` would get no detour, or one annotated ``detour: declined`` would.
Without a catkin build, put mattbot_sim/src, mattbot_navigation/src and path_planning/src on PYTHONPATH.
"""

import argparse
import logging
import os
import sys

from mattbot_sim.detour_check import check_scenario, format_results
from mattbot_sim.scenario import load_scenario


def default_map_dir():
    try:
        import rospkg

        return os.path.join(rospkg.RosPack().get_path("mattbot_mcl"), "map_json")
    except Exception:  # no ROS environment: assume the workspace layout
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mattbot_mcl", "map_json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario")
    ap.add_argument("--map-dir", default=default_map_dir(), help="mattbot_mcl/map_json")
    ap.add_argument("--n-trips", type=float, default=5.0, help="launch arg observe_detour_n_trips")
    ap.add_argument("--margin", type=float, default=0.0, help="launch arg observe_detour_margin_m")
    ap.add_argument("--max-detour", type=float, default=15.0, help="launch arg observe_max_detour_m")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.ros/mattbot_roadmap"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    scenario = load_scenario(args.scenario)
    results = check_scenario(scenario, args.map_dir, args.cache_dir, n_trips=args.n_trips, margin_m=args.margin,
                             max_detour_m=args.max_detour)
    print("%s: N=%g margin=%g m max_detour=%g m, belief 0" % (scenario.name, args.n_trips, args.margin,
                                                              args.max_detour))
    print(format_results(results))

    failures = []
    for r in results:
        want = scenario.detour.get(r.object_id)
        if want == "expected" and not (r.hidden and r.detour_go):
            failures.append("%s: expected a detour, but %s" % (
                r.object_id, "it is visible from the path" if not r.hidden else "the planner would not take one"))
        elif want == "declined" and r.detour_go:
            failures.append("%s: expected no detour, but the planner would take one" % r.object_id)
        if want and r.camera_sees_from_path:
            failures.append("%s: the sim camera can see it from the path (within 5 m), so the robot may "
                            "detect it while patrolling" % r.object_id)
    for f in failures:
        print("FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
