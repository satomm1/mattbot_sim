#!/usr/bin/env python3
"""Plots of a sweep (fleet size, task rate, blocking obstacles or obstacle lifetime) as each method's improvement
over None at every level: the levels on the x axis, one line per method, one panel per map, one figure per metric.

  plot_sweep_gains.py DIR [DIR ...] --factor robots|rate|blocking|lifetime [--out sweep_gains_robots.png]

Each DIR is an experiment run directory (e.g. fleet1, main, fleet5 of every map, as run_all.sh writes them), grouped
by their spec's map and ordered by their spec's level as in plot_sweeps.py. Writes <out stem>_<metric>.png per
metric (know, task_mean, dist), each also as a .jpg at 150 dpi, and the plotted values as <out stem>.csv.

Improvement over None (%) = 100 * (none - condition) / none on the same seeds, so a drop in time or distance is
positive, and the gap between two methods' lines is what the later one adds (the marginal gains of
plot_importance.py). 95 % bootstrap over seeds (all conditions of a level resampled together). Time to know is the
restricted mean (report.RMST_CAP_S); none never drops an obstacle, so its value is the cap. The Oracle is drawn
dashed as the upper bound, except for the time to know (where it is about 100 % at every level).
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from plot_importance import BOOT, METRICS, _value  # noqa: E402
from plot_importance import load as load_run  # noqa: E402
from plot_results import COLORS, FIG_H, FIG_W, FONT_SIZE, GRID, SURFACE, condition_label, map_label, save  # noqa: E402
from plot_sweeps import level_of  # noqa: E402

METHODS = ["opportunistic", "detour", "scout", "oracle"]  # opportunistic, detour, scout: COLORS in this order
X_TITLES = {"robots": "Number of robots", "rate": "Tasks per robot per minute", "blocking": "Blocking obstacles",
            "lifetime": "Obstacle lifetime (min)"}
ORACLE_COLOR = "#77766f"


def gains(data, key, rng):
    """{condition: (%, lo, hi)}: 100 * (none - condition) / none, paired by seed."""
    conds = [c for c in ["none"] + METHODS if c in data]
    seeds = sorted(set.intersection(*[set(data[c]["runs"]["task_mean"]) for c in conds]))
    pick = rng.integers(0, len(seeds), size=(BOOT, len(seeds)))
    out = {}
    for c in conds[1:]:
        gain = lambda ss: 100 * (_value(data, "none", key, ss) - _value(data, c, key, ss)) / _value(data, "none", key, ss)  # noqa: E731
        boots = [gain([seeds[i] for i in row]) for row in pick]
        out[c] = (gain(seeds), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def gather(dirs, factor):
    """[(map, [(level label, {metric: {condition: (%, lo, hi)}}), ...]), ...]: maps in the order given, levels sorted."""
    from mattbot_sim.experiment.spec import load_spec

    rng = np.random.default_rng(0)
    maps = {}
    for d in dirs:
        level = level_of(load_spec(os.path.join(d, "spec.yaml")), factor)
        name, _attrs, data = load_run(d)
        maps.setdefault(name, []).append((level, {key: gains(data, key, rng) for key, _t in METRICS}))
    return [(name, [(lv[1], st) for lv, st in sorted(levels, key=lambda t: t[0][0])]) for name, levels in maps.items()]


def plot(groups, factor, stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": FONT_SIZE})
    paths = []
    for key, title in METRICS:
        # Half the standard canvas: two of these stacked fill FIG_W x FIG_H at the same text size
        fig, axes = plt.subplots(1, len(groups), figsize=(FIG_W, FIG_H / 2.0), facecolor=SURFACE, squeeze=False,
                                 sharey=True)
        for ax, (name, levels) in zip(axes[0], groups):
            ax.set_facecolor(SURFACE)
            x = np.arange(len(levels))
            for k, c in enumerate(METHODS):
                if c == "oracle" and key == "know":
                    continue
                keep = [i for i, (_l, st) in enumerate(levels) if c in st[key]]
                pts = [levels[i][1][key][c] for i in keep]
                err = [[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]]
                style = dict(color=ORACLE_COLOR, ls="--") if c == "oracle" else dict(color=COLORS[k])
                ax.errorbar(x[keep] + (k - 1.5) * 0.05, [p[0] for p in pts], yerr=err, marker="o", lw=2.0, ms=7,
                            capsize=3, elinewidth=1.2, label=condition_label(c), **style)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(map_label(name), pad=4)
            ax.set_xticks(x)
            ax.set_xticklabels([label for label, _st in levels])
            ax.set_xlim(-0.4, len(levels) - 0.6)
            ax.grid(True, axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("Improvement\nover None (%)")
        fig.supxlabel(X_TITLES[factor], fontsize=FONT_SIZE)
        # One header row: the metric (left) and the legend (right)
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
        w.writerow(["metric", "map", "level", "condition", "gain_pct", "lo", "hi"])
        for key, _t in METRICS:
            for name, levels in groups:
                for label, st in levels:
                    for c in METHODS:
                        if c in st[key]:
                            w.writerow([key, name, label, c] + ["%.2f" % v for v in st[key][c]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--factor", required=True, choices=sorted(X_TITLES))
    ap.add_argument("--out", default="", help="PNG path stem (default: sweep_gains_<factor>.png in the first "
                                              "directory's parent; per-metric figures are <stem>_<metric>.png)")
    args = ap.parse_args()
    groups = gather(args.dirs, args.factor)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dirs[0])), "sweep_gains_%s.png" % args.factor)
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    written = plot(groups, args.factor, stem)
    write_table(groups, stem + ".csv")
    print("wrote\n  " + "\n  ".join(written + [stem + ".csv"]))


if __name__ == "__main__":
    main()
