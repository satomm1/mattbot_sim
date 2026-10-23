#!/usr/bin/env python3
"""Plots of experiment results across maps: conditions on the x axis, one figure per metric, one line per map.

  plot_results.py DIR [DIR ...] [--out results.png] [--combined]

Writes <out stem>_<metric>.png per metric (know, task_total, task_mean, dist, dist_ns), each also as a .jpg at
150 dpi; --combined also writes all of them as panels of one figure (<out>, and its .jpg).

Each DIR is an experiment run directory (runs.csv, obstacles.csv, tasks.csv; e.g. one per map from run_batch.sh).
Panels (mean, 95 % CI): time to know an obstacle is gone (restricted mean, report.restricted_mean), total fleet
distance, distance without scout driving, total task time and mean task time (report.add_fleet_metrics). A
condition that never drops an obstacle (none) gets "never" instead of a point (its restricted mean would only be the
cap). The plotted values are also written next to the PNG as <out>.csv.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

ORDER = ["none", "opportunistic", "detour", "scout", "oracle"]
# Display names (the data keeps the condition and map names of the specs)
# Each re-checking condition adds to the previous one ("+"): Active = opportunistic + detours, Scout = + scouts
CONDITION_LABELS = {"none": "None", "opportunistic": "Opportunistic", "detour": "+ Active", "scout": "+ Scout",
                    "oracle": "Oracle"}
X_TITLE = "Re-checking method(s) (each adds to the previous)"
MAP_LABELS = {"current_map": "Y2E2", "grid_office": "Office Grid", "ladder": "Ladder"}


def condition_label(c):
    return CONDITION_LABELS.get(c, c.replace("_", " ").title())


def map_label(m):
    return MAP_LABELS.get(m, m.replace("_", " ").title())
# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface); assigned per map in the
# order the directories are given, never cycled (a fourth map would need its own panel set)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, GRID = "#fcfcfb", "#e4e3df"
PANELS = [  # (key, y-axis label, unit): row 1 times, row 2 distances. Per one-hour run unless noted; the
    # time from an obstacle's removal until the ledger drops it is a restricted mean; task times run from a task's
    # release to the robot's arrival (the mean is per task); excl. scouting leaves out the driving to scout
    # viewpoints and back
    ("know", "Mean removal detection time", "s"),
    ("task_total", "Total task time", "s"),
    ("task_mean", "Mean task time", "s"),
    ("dist", "Fleet distance", "m"),
    ("dist_ns", "Fleet distance excl. scouting", "m"),
]
# Every figure of the plotting scripts (plot_results, plot_sweeps, plot_importance, plot_policy) is drawn on this
# canvas with this text size, so their text reads the same at the same printed size (matplotlib's default is 10 pt)
FIG_W, FIG_H = 10.0, 6.0  # inches
FONT_SIZE = 14  # pt, all text
LABEL_GAP = 0.06  # minimum vertical gap between direct labels (fraction of the panel's y range)


def load(run_dir):
    """(map name, {condition: {metric: (mean, lo, hi)}}) of one run directory."""
    import numpy as np
    import pandas as pd

    from mattbot_sim.experiment.report import RMST_CAP_S, add_fleet_metrics, add_stale_cost, mean_ci, restricted_mean_ci
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(os.path.join(run_dir, "spec.yaml"))
    runs = pd.read_csv(os.path.join(run_dir, "runs.csv"))
    obs = pd.read_csv(os.path.join(run_dir, "obstacles.csv"))
    tasks = pd.read_csv(os.path.join(run_dir, "tasks.csv"))
    runs = add_fleet_metrics(add_stale_cost(runs, obs), tasks)
    rng = np.random.default_rng(0)
    out = {}
    for c in ORDER:
        r = runs[runs["condition"] == c]
        if not len(r):
            continue
        o = obs[(obs["condition"] == c) & (obs["latency_s"].notna() | (obs["censored"] == 1))]
        times = np.where(o["censored"] == 1, o["ghost_s"], o["latency_s"]).astype(float)
        cens = (o["censored"] == 1).to_numpy()
        know = restricted_mean_ci(times, cens, rng) if len(o) else (float("nan"),) * 3
        out[c] = {
            "know": know, "never": bool(len(o)) and bool(cens.all()),
            "dist": mean_ci(r["dist_total_m"], rng), "dist_ns": mean_ci(r["dist_no_scout_m"], rng),
            "task_total": mean_ci(r["task_time_total_s"], rng), "task_mean": mean_ci(r["task_time_mean_s"], rng),
        }
    return spec.map, out, RMST_CAP_S


NOTE = ("Mean over the seeds, 95 %% bootstrap CI (time to know: over obstacles; restricted mean up to %d s). The "
        "intervals are the spread between seeds; differences between conditions on the same seed are much tighter "
        "(paired differences in each report.md). Oracle: the ledger drops an obstacle the moment it is removed.")


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Text and the plot border: matplotlib's defaults, except a larger text size for readability
    plt.rcParams.update({"font.size": FONT_SIZE})
    return plt


def draw_panel(ax, key, title, unit, series, conds):
    """One metric on ax: a line per map over the conditions, CI bars, direct labels at the line ends."""
    import numpy as np

    ax.set_facecolor(SURFACE)
    x = np.arange(len(conds))
    n = len(series)
    offsets = (np.arange(n) - (n - 1) / 2.0) * 0.08  # small dodge so overlapping error bars stay readable
    ends = []
    for k, (name, stats) in enumerate(series):
        col = COLORS[k]
        keep = [c for c in conds if c in stats and not (key == "know" and stats[c].get("never"))]
        xs = [x[conds.index(c)] + offsets[k] for c in keep]
        pts = [stats[c][key] for c in keep]
        m = np.array([p[0] for p in pts], dtype=float)
        lo = np.array([p[1] for p in pts], dtype=float)
        hi = np.array([p[2] for p in pts], dtype=float)
        err = np.vstack([np.where(np.isnan(lo), 0, m - lo), np.where(np.isnan(hi), 0, hi - m)])
        ax.plot(xs, m, "-", color=col, lw=2.0, zorder=2, label=map_label(name))
        ax.errorbar(xs, m, yerr=err, fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=2)
        ax.plot(xs, m, "o", ms=8, mfc=col, mec=SURFACE, mew=1.5, zorder=3)
        # ends.append((xs[-1], m[-1], name))
    if key == "know" and "none" in conds and any(stats.get("none", {}).get("never") for _n, stats in series):
        # ax.annotate("never\n(no checks)", (x[conds.index("none")], 0.92), xycoords=("data", "axes fraction"),
                    # ha="center", va="top", fontsize=9, color=INK_2)
        ax.set_ylim(bottom=0)
    lo_y, hi_y = ax.get_ylim()
    ys = _spread([e[1] for e in ends], LABEL_GAP * (hi_y - lo_y))
    for (xe, _ye, name), yl in zip(ends, ys):  # direct labels: identity is never colour alone
        ax.annotate(map_label(name), (xe, yl), xytext=(10, 0), textcoords="offset points", va="center")
    # The metric names the y axis (no figure title); long names wrap onto two lines
    import textwrap

    ax.set_ylabel("\n".join(textwrap.wrap("%s (%s)" % (title, unit), 34)))
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in conds], rotation=20, ha="right")
    ax.set_xlabel(X_TITLE)
    ax.set_xlim(-0.4, len(conds) - 0.6)  # the same margin on both sides
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _spread(ys, gap):
    """Nudge label heights apart (keeping their order) so no two are closer than gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for j in range(1, len(order)):
        prev, cur = order[j - 1], order[j]
        out[cur] = max(out[cur], out[prev] + gap)
    return out


PNG_DPI, JPG_DPI = 130, 150


def save(fig, png_path, exact=False):
    """The figure as PNG (PNG_DPI) and as a JPG of the same name at JPG_DPI. Returns both paths. exact: keep the
    figure size as it is (the layout must leave room for everything), instead of trimming to the drawn content."""
    jpg_path = os.path.splitext(png_path)[0] + ".jpg"
    bbox = None if exact else "tight"
    fig.savefig(png_path, dpi=PNG_DPI, facecolor=SURFACE, bbox_inches=bbox)
    fig.savefig(jpg_path, dpi=JPG_DPI, facecolor=SURFACE, bbox_inches=bbox, pil_kwargs={"quality": 95})
    return [png_path, jpg_path]


def plot_each(series, cap, stem):
    """One figure per metric: <stem>_<key>.png. Returns the paths."""
    import textwrap

    plt = _style()
    conds = [c for c in ORDER if any(c in s for _m, s in series)]
    paths = []
    for key, title, unit in PANELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
        draw_panel(ax, key, title, unit, series, conds)
        # One row above the plot area, so the legend never covers data
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(series))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        paths += save(fig, "%s_%s.png" % (stem, key))
        plt.close(fig)
    return paths


def plot_combined(series, cap, path):
    """All metrics as panels of one figure (row 1 times, row 2 distances)."""
    import textwrap

    plt = _style()
    conds = [c for c in ORDER if any(c in s for _m, s in series)]
    fig, grid = plt.subplots(2, 3, figsize=(15.5, 9.0), facecolor=SURFACE)
    axes = list(grid[0]) + list(grid[1][:2])
    for ax, (key, title, unit) in zip(axes, PANELS):
        draw_panel(ax, key, title, unit, series, conds)
    note_ax = grid[1][2]
    note_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    note_ax.legend(handles, labels, loc="upper left", frameon=False, title="map")
    note_ax.text(0.0, 0.55, "\n".join(textwrap.wrap(NOTE % cap, 55)), transform=note_ax.transAxes, va="top",
                 )
    fig.tight_layout()
    written = save(fig, path)
    plt.close(fig)
    return written


def write_table(series, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "condition"] + ["%s_%s" % (k, s) for k, _t, _u in PANELS for s in ("mean", "lo", "hi")])
        for name, stats in series:
            for c in ORDER:
                if c in stats:
                    w.writerow([name, c] + ["%.2f" % v for k, _t, _u in PANELS for v in stats[c][k]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="", help="PNG path stem (default: results_by_map.png in the first directory's "
                                              "parent; per-metric figures are <stem>_<metric>.png)")
    ap.add_argument("--combined", action="store_true", help="also all metrics in one figure (<out>)")
    args = ap.parse_args()
    if len(args.dirs) > len(COLORS):
        sys.exit("at most %d run directories (one colour per map)" % len(COLORS))
    series, cap = [], None
    for d in args.dirs:
        name, stats, cap = load(d)
        series.append((name, stats))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dirs[0])), "results_by_map.png")
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    written = plot_each(series, cap, stem)
    if args.combined:
        written += plot_combined(series, cap, out)
    write_table(series, stem + ".csv")
    print("wrote\n  " + "\n  ".join(written + [stem + ".csv"]))


if __name__ == "__main__":
    main()
