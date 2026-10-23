#!/usr/bin/env python3
"""Which re-checking method matters in which setting: a heatmap of each method's marginal gain per setting, and how
the removals were detected per setting.

  plot_importance.py DIR [DIR ...] [--out importance.png]

Each DIR is an experiment run directory, one setting each (e.g. fleet1, main, fleet5, light, busy of every map, as
run_all.sh writes them). Rows are the settings, grouped by their spec's map in the order given.

<out stem>_<metric>.png (know, task_mean, dist; each also as .jpg at 150 dpi): a heatmap, settings x methods. A
method's marginal gain is the improvement over the method before it on the same seeds (Opportunistic over None,
+ Active over Opportunistic, + Scout over + Active; Remaining: the Oracle over + Scout, what is still left), as a
percentage of None's value, so the gains of a row add up to the Oracle's improvement over None. 95 % bootstrap over
seeds (both conditions resampled together); a gain whose interval includes 0 is printed in parentheses. Time to know
is the restricted mean (report.RMST_CAP_S); none never drops an obstacle, so its value is the cap.

<out stem>_sources.png: how the + Scout condition (every method on) detected the removed obstacles it knew of, as
shares: opportunistic (seen in passing, at a task's start or goal), detour, scout, or not detected within the run.

The plotted values are also written as <out stem>.csv and <out stem>_sources.csv.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from plot_results import COLORS, FIG_H, FIG_W, FONT_SIZE, SURFACE, map_label, save  # noqa: E402

# (column label, previous condition, condition)
LAYERS = [("Opportunistic", "none", "opportunistic"), ("+ Active", "opportunistic", "detour"),
          ("+ Scout", "detour", "scout"), ("Remaining\n(to Oracle)", "scout", "oracle")]
METRICS = [("know", "Mean removal detection time"), ("task_mean", "Mean task time"), ("dist", "Fleet distance")]
RUN_COLUMNS = {"task_mean": "task_time_mean_s", "dist": "dist_total_m", "check": "dist_check_m_per_h"}
SOURCES = [("Opportunistic", ("OPPORTUNISTIC", "AT_START", "AT_GOAL"), COLORS[0]), ("Detour", ("DETOUR",), COLORS[1]),
           ("Scout", ("SCOUT",), COLORS[2]), ("Not detected", (), "#b5b4ae")]
SOURCE_CONDITION = "scout"
BOOT = 500


def setting_attrs(spec):
    """A setting's attributes, as (key, label text) in label order."""
    robots = len(spec.robots)
    blocking = int(spec.obstacle_types.get("blocking", {}).get("count", 0))
    lo, hi = (float(v) for v in spec.obstacles["lifetime_s"])
    return [("robots", "%d robot%s" % (robots, "" if robots == 1 else "s")),
            ("rate", "%.2f tasks/robot/min" % (spec.release_rate_per_min / max(robots, 1))),
            ("blocking", "%d blocking" % blocking),
            ("lifetime", "%g-%g min lifetime" % (lo / 60.0, hi / 60.0))]


def setting_labels(attrs):
    """Row labels from the attributes that differ between the settings (robots and rate when none do)."""
    keys = [k for k, _t in attrs[0] if len({dict(a)[k] for a in attrs}) > 1] or ["robots", "rate"]
    return [", ".join(t for k, t in a if k in keys) for a in attrs]


def load(run_dir):
    """(map, setting attributes, {condition: {"runs": {metric: {seed: value}}, "obs": {seed: (times, censored)}, "by": [...]}})"""
    import pandas as pd

    from mattbot_sim.experiment.report import add_fleet_metrics, add_stale_cost
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(os.path.join(run_dir, "spec.yaml"))
    runs = pd.read_csv(os.path.join(run_dir, "runs.csv"))
    obs = pd.read_csv(os.path.join(run_dir, "obstacles.csv"))
    runs = add_fleet_metrics(add_stale_cost(runs, obs), pd.read_csv(os.path.join(run_dir, "tasks.csv")))
    data = {}
    for c, r in runs.groupby("condition"):
        o = obs[(obs["condition"] == c) & (obs["latency_s"].notna() | (obs["censored"] == 1))]
        per_seed = {}
        for s, os_ in o.groupby("seed"):
            cens = (os_["censored"] == 1).to_numpy()
            per_seed[s] = (np.where(cens, os_["ghost_s"], os_["latency_s"]).astype(float), cens)
        data[c] = {"runs": {k: r.set_index("seed")[col].to_dict() for k, col in RUN_COLUMNS.items()}, "obs": per_seed,
                   "by": ["NOT_DETECTED" if cz == 1 else b for b, cz in zip(o["dropped_by"], o["censored"])]}
    return spec.map, setting_attrs(spec), data


def _value(data, c, key, seeds):
    """The metric of condition c over the given seeds (with repeats)."""
    from mattbot_sim.experiment.report import restricted_mean

    if key == "know":
        empty = (np.zeros(0), np.zeros(0, dtype=bool))
        parts = [data[c]["obs"].get(s, empty) for s in seeds]
        return restricted_mean(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    vals = data[c]["runs"][key]
    return float(np.mean([vals[s] for s in seeds]))


def marginal_gains(data, key, rng):
    """{layer label: (%, lo, hi)}: 100 * (previous - condition) / none, paired by seed."""
    conds = [c for _l, a, b in LAYERS for c in (a, b)]
    seeds = sorted(set.intersection(*[set(data[c]["runs"]["task_mean"]) for c in set(conds) if c in data]))
    pick = rng.integers(0, len(seeds), size=(BOOT, len(seeds)))
    out = {}
    for label, prev, cur in LAYERS:
        if prev not in data or cur not in data:
            continue
        gain = lambda ss: 100 * (_value(data, prev, key, ss) - _value(data, cur, key, ss)) / _value(data, "none", key, ss)  # noqa: E731
        boots = [gain([seeds[i] for i in row]) for row in pick]
        out[label] = (gain(seeds), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def gather(dirs):
    """[(map, [(row label, data), ...]), ...]: maps in the order given, settings in the order given. The labels name
    only what differs between the settings (of every map)."""
    loaded = [load(d) for d in dirs]
    labels = setting_labels([attrs for _m, attrs, _d in loaded])
    maps = {}
    for (name, _attrs, data), label in zip(loaded, labels):
        maps.setdefault(name, []).append((label, data))
    return list(maps.items())


def _rows(groups):
    """Row labels top to bottom, and (map, first row, last row) per map."""
    labels, spans = [], []
    for name, settings in groups:
        spans.append((name, len(labels), len(labels) + len(settings) - 1))
        labels += [label for label, _d in settings]
    return labels, spans


def _map_brackets(ax, spans, labels):
    """Map names left of the row labels (clear of the longest one), and a line between the maps."""
    offset = -(0.58 * FONT_SIZE * max(len(lab) for lab in labels) + 1.5 * FONT_SIZE)
    for k, (name, first, last) in enumerate(spans):
        ax.annotate(map_label(name), xy=(0, (first + last) / 2.0), xycoords=("axes fraction", "data"),
                    xytext=(offset, 0), textcoords="offset points", ha="center", va="center", rotation=90,
                    fontweight="bold")
        if k:
            ax.axhline(first - 0.5, color="black", lw=1.5)


def plot_heatmaps(groups, stem, rng):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({"font.size": FONT_SIZE})
    labels, spans = _rows(groups)
    table, paths = [], []
    for key, title in METRICS:
        cells = [marginal_gains(data, key, rng) for _name, settings in groups for _l, data in settings]
        z = np.array([[cell.get(lab, (np.nan,) * 3)[0] for lab, _a, _b in LAYERS] for cell in cells])
        row_maps = [name for name, settings in groups for _ in settings]
        for m, label, cell in zip(row_maps, labels, cells):
            for lab, _a, _b in LAYERS:
                if lab in cell:
                    table.append([key, m, label, lab.replace("\n", " ")] + ["%.2f" % v for v in cell[lab]])
        lim = np.nanmax(np.abs(z)) or 1.0
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
        im = ax.imshow(z, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i, cell in enumerate(cells):
            for j, (lab, _a, _b) in enumerate(LAYERS):
                if lab not in cell:
                    continue
                m, lo, hi = cell[lab]
                txt = "%.1f" % m if (lo > 0 or hi < 0) else "(%.1f)" % m
                ax.text(j, i, txt, ha="center", va="center", color="white" if abs(m) > 0.6 * lim else "black")
        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([lab for lab, _a, _b in LAYERS])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(len(LAYERS) - 1.5, color="black", lw=1.0, ls="--")  # the methods | what is left
        _map_brackets(ax, spans, labels)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("%s:\nmarginal improvement (%% of None)" % title)
        fig.tight_layout()
        paths += save(fig, "%s_%s.png" % (stem, key))
        plt.close(fig)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "map", "setting", "method", "gain_pct", "lo", "hi"])
        w.writerows(table)
    return paths + [stem + ".csv"]


def plot_sources(groups, stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": FONT_SIZE})
    labels, spans = _rows(groups)
    shares = []
    for _name, settings in groups:
        for _label, data in settings:
            by = data[SOURCE_CONDITION]["by"]
            counts = [sum(1 for b in by if (b in kinds if kinds else b == "NOT_DETECTED")) for _s, kinds, _c in SOURCES]
            shares.append((100.0 * np.array(counts) / max(len(by), 1), len(by)))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
    left = np.zeros(len(labels))
    y = np.arange(len(labels))
    for k, (src, _kinds, col) in enumerate(SOURCES):
        w = np.array([s[k] for s, _n in shares])
        ax.barh(y, w, left=left, color=col, edgecolor=SURFACE, linewidth=2, label=src)
        for yi, (li, wi) in enumerate(zip(left, w)):
            if wi >= 6:
                ax.text(li + wi / 2, yi, "%.0f" % wi, ha="center", va="center")
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Removed obstacles the fleet knew of, + Scout condition (%)")
    _map_brackets(ax, spans, labels)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(SOURCES), frameon=False)
    fig.tight_layout()
    paths = save(fig, stem + "_sources.png")
    plt.close(fig)
    with open(stem + "_sources.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "setting", "obstacles"] + [s for s, _k, _c in SOURCES])
        row_maps = [name for name, settings in groups for _ in settings]
        for m, label, (s, n) in zip(row_maps, labels, shares):
            w.writerow([m, label, n] + ["%.1f" % v for v in s])
    return paths + [stem + "_sources.csv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="", help="PNG path stem (default: importance.png in the first directory's parent)")
    args = ap.parse_args()
    groups = gather(args.dirs)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dirs[0])), "importance.png")
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    written = plot_heatmaps(groups, stem, np.random.default_rng(0)) + plot_sources(groups, stem)
    print("wrote\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
