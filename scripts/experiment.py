#!/usr/bin/env python3
"""Simulation experiments on the fast simulator (no ROS): re-checking strategies on the same seeded instances.

  experiment.py generate experiments/pilot.yaml [--seed 0] [--out DIR]     # candidates (cached), an instance, a PNG
  experiment.py run experiments/pilot.yaml [--out DIR] [--jobs 4] [--seeds 5 | --seeds 0,3,7] [--conditions a,b]
  experiment.py report DIR [--baseline none] [--latency-baseline opportunistic]
  experiment.py replay DIR --condition scout --seed 0 [--gif run.gif] [--png run.png] [-v]

See mattbot_sim/README.md "Experiments" for the spec format and the metrics.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
PKG = os.path.normpath(os.path.join(HERE, ".."))


def parse_seeds(text):
    if not text:
        return None
    return [int(s) for s in text.split(",")] if "," in text else list(range(int(text)))


def default_out(spec):
    return os.path.join(PKG, "results", "exp_%s_%s" % (spec.name, time.strftime("%Y%m%d_%H%M%S")))


def cmd_generate(args):
    from mattbot_sim.experiment.generate import make_instance, plot_instance, setup, write_scenario
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(args.spec)
    out = args.out or os.path.join(PKG, "results", "exp_%s_generate" % spec.name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()  # wall clock: progress
    ctx, landmarks, cands, conn = setup(spec, progress=lambda k, n: (
        print("  candidates %d/%d" % (k, n), flush=True) if k % 50 == 0 else None))
    for name, (x, y, _th) in landmarks.items():
        ox, oy, _ = spec.landmarks[name]
        if abs(ox - x) > 1e-6 or abs(oy - y) > 1e-6:
            print("landmark %s snapped from (%.2f, %.2f) to (%.2f, %.2f)" % (name, ox, oy, x, y))
    usable = [c for c in cands if c["I_o"] >= float(spec.obstacles["min_importance_m"])]
    print("%d candidate placements, %d with I_o >= %g m (%.0f s)" % (
        len(cands), len(usable), spec.obstacles["min_importance_m"], time.time() - t0))  # wall clock: progress
    inst = make_instance(spec, args.seed, cands, landmarks, conn)
    base = os.path.join(out, "seed%d" % args.seed)
    with open(base + ".json", "w") as f:
        json.dump(inst, f, indent=1)
    write_scenario(spec, inst, base + ".yaml")
    plot_instance(spec, ctx, landmarks, cands, inst, base + ".png")
    for ob in inst["obstacles"]:
        print("  %s %s at (%.2f, %.2f) I_o=%.1f m, %.0f-%.0f s" % (ob["id"], ob["class"], ob["x"], ob["y"], ob["I_o"],
                                                                  ob["appear_s"], ob["remove_s"]))
    print("%d tasks over %.0f s; wrote %s.{json,yaml,png}" % (len(inst["tasks"]), inst["tasks"][-1]["release_s"], base))


def cmd_run(args):
    from mattbot_sim.experiment.report import report
    from mattbot_sim.experiment.runner import run
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(args.spec)
    out = args.out or default_out(spec)
    conditions = args.conditions.split(",") if args.conditions else None
    t0 = time.time()  # wall clock: progress

    def progress(k, n, row):
        print("[%d/%d] %s seed %d: %d/%d tasks, %.0f m, %d dropped / %d censored (%.0f s wall, %.0f s total)" % (
            k, n, row["condition"], row["seed"], row["tasks_done"], row["tasks"], row["dist_total_m"],
            row["obstacles_dropped"], row["obstacles_censored"], row["wall_s"], time.time() - t0), flush=True)  # wall clock

    run(spec.path, out, conditions, parse_seeds(args.seeds), args.jobs, progress=progress)
    print(report(out, args.baseline, args.latency_baseline))
    print("results in", out)


def cmd_report(args):
    from mattbot_sim.experiment.report import report

    print(report(args.dir, args.baseline, args.latency_baseline))


def cmd_replay(args):
    from mattbot_sim.experiment.orchestrator import TaskOrchestrator
    from mattbot_sim.experiment.spec import load_spec
    from mattbot_sim.fast import viz
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    spec = load_spec(os.path.join(args.dir, "spec.yaml") if os.path.isdir(args.dir) else args.dir)
    inst_dir = os.path.join(args.dir, "instances") if os.path.isdir(args.dir) else None
    if inst_dir is None or not os.path.exists(os.path.join(inst_dir, "seed%d.json" % args.seed)):
        sys.exit("no instance seed%d in %s (run the experiment first)" % (args.seed, args.dir))
    with open(os.path.join(inst_dir, "seed%d.json" % args.seed)) as f:
        inst = json.load(f)
    settings = spec.params_of(args.condition)
    params = FastParams().update(settings)
    params._explicit = dict(settings)
    sim = FastSim(os.path.join(inst_dir, "seed%d.yaml" % args.seed), params=params, seed=args.seed,
                  orchestrator=TaskOrchestrator(inst["tasks"], dwell_s=spec.tasks["dwell_s"]),
                  duration=spec.max_duration_s, verbose=args.verbose)
    if args.gif or args.png:
        sim.frames, sim.frame_every = [], args.frame_every
    sim.run()
    print("%s seed %d: %.0f s simulated" % (args.condition, args.seed, sim.now))
    if args.gif:
        viz.render_gif(sim, sim.frames, args.gif)
        print("wrote", args.gif)
    if args.png:
        viz.render_png(sim, sim.frames, args.png)
        print("wrote", args.png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("generate")
    p.add_argument("spec")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_generate)
    p = sub.add_parser("run")
    p.add_argument("spec")
    p.add_argument("--out", default="")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seeds", default="", help="a count (0..n-1) or a comma list; default: the spec's")
    p.add_argument("--conditions", default="", help="comma list; default: every condition of the spec")
    p.add_argument("--baseline", default=None, help="distance baseline (default: the first condition)")
    p.add_argument("--latency-baseline", default=None)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.add_argument("--baseline", default=None)
    p.add_argument("--latency-baseline", default=None)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("replay")
    p.add_argument("dir", help="an experiment run directory")
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", default="")
    p.add_argument("--png", default="")
    p.add_argument("--frame-every", type=float, default=4.0)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_replay)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
