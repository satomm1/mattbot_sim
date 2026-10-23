#!/usr/bin/env python3
"""Plots of a sweep (fleet size, task rate, blocking obstacles or obstacle lifetime) as the improvement from one level to the next: the conditions on the
x axis, one line per interval (e.g. 1 -> 3 robots, 3 -> 5 robots), one panel per map, one figure per metric.

  plot_sweeps.py DIR [DIR ...] --factor robots|rate|blocking|lifetime [--out sweep_robots.png]

Each DIR is an experiment run directory (e.g. fleet1, main, fleet5 of every map, as run_all.sh writes them). The
directories are grouped by their spec's map (one panel each, in the order given) and ordered by their spec's level:
the number of robots, the task rate per robot per minute, the number of blocking obstacles, or the obstacles'
lifetime range (ordered by its shortest lifetime). Writes <out stem>_<metric>.png per metric (know,
task_total, task_mean, dist, dist_ns), each also as a .jpg at 150 dpi, and the plotted values as <out stem>.csv.

Improvement (%) = 100 * (previous level - this level) / previous level, so a drop in time or distance is positive.
The levels are separate experiments (their instances differ), so the 95 % interval resamples each level on its own:
seeds for the run metrics, obstacles for the time to know (a restricted mean, report.RMST_CAP_S). The none condition
never drops an obstacle, so it has no time to know; the oracle's is about 0.5 s at every level (the
ledger update after a removal), so its change is only noise and is left out too.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from plot_results import (  # noqa: E402
    COLORS, FIG_H, FIG_W, FONT_SIZE, GRID, ORDER, PANELS, SURFACE, X_TITLE, condition_label, map_label, save,
)

FACTORS = {"robots": " robots", "rate": " tasks/robot/min", "blocking": " blocking", "lifetime": " min lifetime"}  # unit
RUN_COLUMNS = {"task_total": "task_time_total_s", "task_mean": "task_time_mean_s", "dist": "dist_total_m",
               "dist_ns": "dist_no_scout_m"}
BOOT = 1000


def level_of(spec, factor):
    """(sort key, label) of a spec's level of the factor."""
    if factor == "robots":
        return len(spec.robots), "%d" % len(spec.robots)
    if factor == "rate":
        rate = spec.release_rate_per_min / max(len(spec.robots), 1)
        return rate, "%.2f" % rate
    if factor == "blocking":
        n = int(spec.obstacle_types.get("blocking", {}).get("count", 0))
        return n, "%d" % n
    lo, hi = (float(v) for v in spec.obstacles["lifetime_s"])
    return lo, "%g-%g" % (lo / 60.0, hi / 60.0)


def load(run_dir, factor):
    """(map, (sort key, label), {condition: {metric: samples}}): per-seed values of the run metrics, (times, censored) of the
    obstacles for the time to know."""
    import pandas as pd

    from mattbot_sim.experiment.report import add_fleet_metrics, add_stale_cost
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(os.path.join(run_dir, "spec.yaml"))
    level = level_of(spec, factor)
    runs = pd.read_csv(os.path.join(run_dir, "runs.csv"))
    obs = pd.read_csv(os.path.join(run_dir, "obstacles.csv"))
    tasks = pd.read_csv(os.path.join(run_dir, "tasks.csv"))
    runs = add_fleet_metrics(add_stale_cost(runs, obs), tasks)
    out = {}
    for c in ORDER:
        r = runs[runs["condition"] == c]
        if not len(r):
            continue
        o = obs[(obs["condition"] == c) & (obs["latency_s"].notna() | (obs["censored"] == 1))]
        cens = (o["censored"] == 1).to_numpy()
        times = np.where(cens, o["ghost_s"], o["latency_s"]).astype(float)
        out[c] = {key: r[col].to_numpy(dtype=float) for key, col in RUN_COLUMNS.items()}
        out[c]["know"] = None if (c == "oracle" or not len(o) or cens.all()) else (times, cens)
    return spec.map, level, out


def improvement(prev, cur, key, rng):
    """(%, lo, hi) of 100 * (1 - cur / prev), each level resampled on its own."""
    from mattbot_sim.experiment.report import restricted_mean

    if key == "know":
        (tp, cp), (tc, cc) = prev, cur
        stat = lambda t, c: restricted_mean(t, c)  # noqa: E731
        pct = 100 * (1 - stat(tc, cc) / stat(tp, cp))
        boots = []
        for _ in range(BOOT // 2):
            i = rng.integers(0, len(tp), len(tp))
            j = rng.integers(0, len(tc), len(tc))
            boots.append(100 * (1 - stat(tc[j], cc[j]) / stat(tp[i], cp[i])))
    else:
        pct = 100 * (1 - cur.mean() / prev.mean())
        bp = rng.choice(prev, size=(BOOT, len(prev))).mean(axis=1)
        bc = rng.choice(cur, size=(BOOT, len(cur))).mean(axis=1)
        boots = 100 * (1 - bc / bp)
    return pct, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def gather(dirs, factor):
    """[(map, [(interval label, {condition: {metric: (%, lo, hi)}}), ...]), ...]: maps in the order given."""
    maps = {}
    for d in dirs:
        name, level, samples = load(d, factor)
        maps.setdefault(name, []).append((level, samples))
    rng = np.random.default_rng(0)
    groups = []
    for name, levels in maps.items():
        levels.sort(key=lambda t: t[0])
        intervals = []
        for (xp, sp), (xc, sc) in zip(levels, levels[1:]):
            stats = {}
            for c in ORDER:
                if c in sp and c in sc:
                    stats[c] = {key: improvement(sp[c][key], sc[c][key], key, rng)
                                if sp[c][key] is not None and sc[c][key] is not None else None
                                for key, _t, _u in PANELS}
            intervals.append(("%s → %s" % (xp[1], xc[1]), stats))
        groups.append((name, intervals))
    return groups


def plot(groups, factor, stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    plt.rcParams.update({"font.size": FONT_SIZE})
    conds = [c for c in ORDER if any(c in st for _m, iv in groups for _l, st in iv)]
    unit = FACTORS[factor]
    paths = []
    for key, title, _unit in PANELS:
        # Half the standard canvas: two of these stacked fill FIG_W x FIG_H at the same text size
        fig, axes = plt.subplots(1, len(groups), figsize=(FIG_W, FIG_H / 2.0), facecolor=SURFACE, squeeze=False,
                                 sharey=True)
        for ax, (name, intervals) in zip(axes[0], groups):
            ax.set_facecolor(SURFACE)
            x = np.arange(len(conds))
            for k, (label, stats) in enumerate(intervals):
                keep = [i for i, c in enumerate(conds) if c in stats and stats[c][key] is not None]
                pts = [stats[conds[i]][key] for i in keep]
                m = [p[0] for p in pts]
                err = [[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]]
                ax.errorbar(x[keep], m, yerr=err, fmt="-o", color=COLORS[k], lw=2.0, ms=7, capsize=3, elinewidth=1.2,
                            label="%s%s" % (label, unit))
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(map_label(name), pad=4)
            ax.set_xticks(x)
            ax.set_xticklabels([condition_label(c) for c in conds], rotation=35, ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.4, len(conds) - 0.6)
            ax.grid(True, axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
        # One header row: the metric (left) and the legend (right)
        axes[0][0].set_ylabel("Improvement vs.\nprevious level (%)")
        fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.88))
        fig.text(0.01, 0.995, title, ha="left", va="top")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=len(labels), frameon=False,
                   handlelength=1.5, columnspacing=1.0)
        paths += save(fig, "%s_%s.png" % (stem, key))
        plt.close(fig)
    return paths


def write_table(groups, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "interval", "condition"] + ["%s_pct_%s" % (k, s) for k, _t, _u in PANELS
                                                       for s in ("mean", "lo", "hi")])
        for name, intervals in groups:
            for label, stats in intervals:
                for c in ORDER:
                    if c in stats:
                        w.writerow([name, label, c] + [
                            "%.2f" % v if stats[c][k] is not None else "" for k, _t, _u in PANELS
                            for v in (stats[c][k] or (0, 0, 0))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--factor", required=True, choices=sorted(FACTORS))
    ap.add_argument("--out", default="", help="PNG path stem (default: sweep_<factor>.png in the first directory's "
                                              "parent; per-metric figures are <stem>_<metric>.png)")
    args = ap.parse_args()
    groups = gather(args.dirs, args.factor)
    if max(len(iv) for _m, iv in groups) > len(COLORS):
        sys.exit("at most %d intervals (one colour each)" % len(COLORS))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dirs[0])), "sweep_%s.png" % args.factor)
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    written = plot(groups, args.factor, stem)
    write_table(groups, stem + ".csv")
    print("wrote\n  " + "\n  ".join(written + [stem + ".csv"]))


if __name__ == "__main__":
    main()
