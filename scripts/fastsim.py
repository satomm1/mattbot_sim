#!/usr/bin/env python3
"""Fast, ROS-free simulator (mattbot_sim.fast): same scenarios, scoring and expectations as the ROS sim.

  fastsim.py run detour_present --set observe_detour=true [--seed 3] [--duration 300] [--out DIR] [-v]
  fastsim.py run fleet_detour_handoff --gif handoff.gif --png handoff.png [--frame-every 2]
  fastsim.py sweep detour_removed --grid observe_detour_n_trips=2,5,10 --set observe_detour=true \\
      --seeds 5 --jobs 4 --out sweep.csv            # also writes sweep_aggregate.csv
  fastsim.py compare detour_present results/detour_present_*.json [--set ...]   # vs ROS sim summaries

Scenarios are names in mattbot_sim/scenarios or paths. --set takes sim.launch arg names (FastParams), and the
scenario's launch_args apply too (--set wins). Exit code: run 0 if every robot's expectations passed.
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mattbot_sim.fast.params import FastParams  # noqa: E402

PKG = os.path.normpath(os.path.join(HERE, ".."))

COMPARE_FIELDS = [
    "ledger_objects_added", "correct_removals", "false_removals", "stale_ledger_objects",
    "observation_stops_by_kind.OPPORTUNISTIC.ENDED", "observation_stops_by_kind.DETOUR.ENDED",
    "observation_outcomes.PRESENT", "observation_outcomes.ABSENT", "observation_outcomes.INCONCLUSIVE",
    "removal_latency_s.mean", "detours_started", "detours_abandoned", "goal_resends",
]


def scenario_path(name):
    if os.path.isfile(name):
        return os.path.abspath(name)
    path = os.path.join(PKG, "scenarios", name if name.endswith(".yaml") else name + ".yaml")
    if not os.path.isfile(path):
        sys.exit("no scenario %s" % name)
    return path


def parse_sets(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit("--set/--grid take name=value, got %r" % item)
        k, v = item.split("=", 1)
        if k not in FastParams.names():
            sys.exit("unknown parameter %r (see mattbot_sim/fast/params.py)" % k)
        out[k] = v
    return out


def make_params(sets):
    p = FastParams().update(sets)
    p._explicit = dict(sets)
    return p


def print_summary(summaries):
    ok = True
    for rid, s in summaries.items():
        passed = s.get("expectations_passed", True)
        ok &= bool(passed)
        print("robot %s: %s  (%.0f s simulated in %.1f s wall; ledger added %d, removals %d correct / %d false, "
              "stale %d, stops %s)" % (rid, "PASSED" if passed else "FAILED", s["run_s"], s["run_wall_s"],
                                       s["ledger_objects_added"], s["correct_removals"], s["false_removals"],
                                       len(s["stale_ledger_objects"]), s["observation_stops"]))
        for c in s.get("expectations", []):
            print("    %s  %s %s %g (got %s)" % ("PASS" if c["ok"] else "FAIL", c["key"], c["op"], c["expected"],
                                               c["actual"]))
    return ok


def cmd_run(args):
    from mattbot_sim.fast.orchestrator import load_orchestrator
    from mattbot_sim.fast.sim import FastSim

    orch = load_orchestrator(args.orchestrator) if args.orchestrator else None
    sim = FastSim(scenario_path(args.scenario), params=make_params(parse_sets(args.set)), seed=args.seed,
                  orchestrator=orch, duration=args.duration, verbose=args.verbose)
    if args.gif or args.png:
        sim.frames, sim.frame_every = [], args.frame_every
    summaries = sim.run()
    ok = print_summary(summaries)
    if args.gif or args.png:
        from mattbot_sim.fast import viz

        if args.gif:
            viz.render_gif(sim, sim.frames, args.gif)
            print("wrote", args.gif)
        if args.png:
            viz.render_png(sim, sim.frames, args.png)
            print("wrote", args.png)
    if args.out:
        for path in sim.write_results(args.out):
            print("wrote", path)
    return 0 if ok else 1


def parse_grid(items):
    grid = {}
    for k, v in parse_sets(items).items():
        grid[k] = [x for x in v.split(",") if x != ""]
    return grid


def cmd_sweep(args):
    from mattbot_sim.fast import sweep as sw

    grid = parse_grid(args.grid)
    fixed = parse_sets(args.set)
    for k, v in fixed.items():
        grid.setdefault(k, [v])
    seeds = list(range(args.seed, args.seed + args.seeds))
    t0 = time.time()  # wall clock: progress report
    n = 1
    for v in grid.values():
        n *= len(v)
    print("%d parameter sets x %d seeds = %d runs, %d jobs" % (n, len(seeds), n * len(seeds), args.jobs))

    def progress(done, total):
        print("\r%d/%d runs (%.0f s)" % (done, total, time.time() - t0), end="", flush=True)  # wall clock: progress

    rows = sw.sweep(scenario_path(args.scenario), grid, seeds, jobs=args.jobs, duration=args.duration,
                    progress=progress)
    print()
    sw.write_csv(args.out, rows)
    agg_path = os.path.splitext(args.out)[0] + "_aggregate.csv"
    sw.write_csv(agg_path, sw.aggregate(rows, sorted(grid)))
    print("wrote %s (%d rows) and %s" % (args.out, len(rows), agg_path))
    return 0


def summary_value(s, path):
    from mattbot_sim.scoring import summary_value as sv

    try:
        return sv(s, path)
    except (KeyError, TypeError):
        return None


def cmd_compare(args):
    from mattbot_sim.fast.sim import FastSim

    sim = FastSim(scenario_path(args.scenario), params=make_params(parse_sets(args.set)), seed=args.seed,
                  duration=args.duration)
    fast = sim.run()
    ros = []
    for pattern in args.ros:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                ros.append((os.path.basename(path), json.load(f)["summary"]))
    if not ros:
        sys.exit("no ROS results matched %s" % args.ros)
    for name, rs in ros:
        rid = rs.get("robot_id", next(iter(fast)))
        fs = fast.get(rid) or next(iter(fast.values()))
        print("\n%s (robot %s): ROS %.0f s vs fast %.0f s simulated" % (name, rid, rs.get("run_s", 0), fs["run_s"]))
        print("  %-48s %10s %10s" % ("field", "ROS", "fast"))
        fmt = lambda v: "%.1f" % v if isinstance(v, float) else str(v)  # noqa: E731
        for field in COMPARE_FIELDS:
            print("  %-48s %10s %10s" % (field, fmt(summary_value(rs, field)), fmt(summary_value(fs, field))))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sweep", "compare"):
        p = sub.add_parser(name)
        p.add_argument("scenario")
        p.add_argument("--set", action="append", default=[], help="name=value (FastParams / sim.launch arg)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--duration", type=float, default=None, help="simulated s (default: the scenario's)")
        if name == "run":
            p.add_argument("--out", default="", help="write sim_monitor-style results JSON here")
            p.add_argument("--orchestrator", default="", help="package.module:Class")
            p.add_argument("-v", "--verbose", action="store_true", help="print the event log")
            p.add_argument("--gif", default="", help="write an animation of the run (GIF)")
            p.add_argument("--png", default="", help="write an overview image: tracks and checks (PNG)")
            p.add_argument("--frame-every", type=float, default=2.0, help="simulated s between GIF frames")
        if name == "sweep":
            p.add_argument("--grid", action="append", default=[], help="name=v1,v2,...")
            p.add_argument("--seeds", type=int, default=1)
            p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
            p.add_argument("--out", default="sweep.csv")
        if name == "compare":
            p.add_argument("ros", nargs="+", help="ROS sim results JSON files (globs ok)")
    args = ap.parse_args()
    sys.exit({"run": cmd_run, "sweep": cmd_sweep, "compare": cmd_compare}[args.cmd](args))


if __name__ == "__main__":
    main()
