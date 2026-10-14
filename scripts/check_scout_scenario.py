#!/usr/bin/env python3
"""Offline check of a scenario's scout targets (no ROS master needed; see mattbot_sim/detour_check.py).

  rosrun mattbot_sim check_scout_scenario.py $(rospack find mattbot_sim)/scenarios/scout_present.yaml \
      [--n-trips 5] [--margin 0] [--max-scout 30] [--fixed-s 0] [--per-s-away 0] [--min-idle 10]

Prints, per object and per pose where the robot sits idle long enough (start without waypoints, long
pauses, the end of a non-looping patrol), the scout the observation planner would decide on with the
object's belief fully decayed. Settings default to the scenario's launch_args (sim.launch names and
defaults); options override them. Exits 1 if an object annotated ``scout: expected`` would get no scout,
one annotated ``scout: declined`` would, or the robot could see an annotated object from an idle pose.
Without a catkin build, put mattbot_sim/src, mattbot_navigation/src and path_planning/src on PYTHONPATH.
"""

import argparse
import logging
import os
import sys

from mattbot_sim.detour_check import check_scout, format_scout_results, scout_failures, scout_params_of
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
    ap.add_argument("--n-trips", type=float, help="launch arg observe_scout_n_trips")
    ap.add_argument("--margin", type=float, help="launch arg observe_scout_margin_m")
    ap.add_argument("--max-scout", type=float, help="launch arg observe_max_scout_m")
    ap.add_argument("--fixed-s", type=float, help="launch arg observe_scout_fixed_s")
    ap.add_argument("--per-s-away", type=float, help="launch arg observe_scout_per_s_away")
    ap.add_argument("--one-way", action="store_true", help="observe_scout_round_trip:=false")
    ap.add_argument("--min-idle", type=float, help="launch arg observe_scout_min_idle_s")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.ros/mattbot_roadmap"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    scenario = load_scenario(args.scenario)
    p = scout_params_of(scenario, observe_scout_n_trips=args.n_trips, observe_scout_margin_m=args.margin,
                        observe_max_scout_m=args.max_scout, observe_scout_fixed_s=args.fixed_s,
                        observe_scout_per_s_away=args.per_s_away, observe_scout_min_idle_s=args.min_idle,
                        observe_scout_round_trip=False if args.one_way else None)
    results = check_scout(scenario, args.map_dir, args.cache_dir, params=p)
    print("%s: N=%g margin=%g m max_scout=%g m, %s, fixed_s=%g s, per_s_away=%g, idle >= %g s, belief 0" % (
        scenario.name, p.observe_scout_n_trips, p.observe_scout_margin_m, p.observe_max_scout_m,
        "out and back" if p.observe_scout_round_trip else "one way", p.observe_scout_fixed_s,
        p.observe_scout_per_s_away, p.observe_scout_min_idle_s))
    print(format_scout_results(results))
    failures = scout_failures(scenario, results, args.map_dir)
    for f in failures:
        print("FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
