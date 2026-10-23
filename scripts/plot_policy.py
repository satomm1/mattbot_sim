#!/usr/bin/env python3
"""Plots of the policy sweep: what more aggressive detours and scouts buy, and the driving they cost.

  plot_policy.py DIR [DIR ...] [--out policy.png]

Each DIR is a policy-sweep run directory (experiments/rechecking/policy.yaml and maps/<map>_policy.yaml: the base
conditions plus detour_n2 / scout_n2 / detour_n10 / scout_n10 / detour_n20 / scout_n20), one column per map in the
order given. Writes <out stem>_<metric>.png per metric (know, task_mean, dist), each also as a .jpg at 150 dpi, and
the plotted values as <out stem>.csv.

The x axis is the level (n_trips 2 / 5 / 10 / 20, the larger levels also with longer distance limits; see the spec),
one line per family (+ Active, + Scout). Top row: improvement over None (% of None, paired by seed, 95 % bootstrap
over seeds; time to know as the restricted mean, report.RMST_CAP_S), with Opportunistic (no check trips) and the
Oracle as reference lines. Bottom row: driving on check trips (detours and scouts, m per hour; mean, 95 % bootstrap
over seeds), the cost of that improvement.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from plot_importance import BOOT, METRICS, _value, load  # noqa: E402
from plot_results import COLORS, FIG_H, FIG_W, FONT_SIZE, GRID, SURFACE, map_label, save  # noqa: E402

# (family label, colour, [(level label, condition), ...]); colours as the detection sources in plot_importance
FAMILIES = [("+ Active", COLORS[1], [("n2", "detour_n2"), ("n5", "detour"), ("n10", "detour_n10"),
                                     ("n20", "detour_n20")]),
            ("+ Scout", COLORS[2], [("n2", "scout_n2"), ("n5", "scout"), ("n10", "scout_n10"), ("n20", "scout_n20")])]
REFERENCE = "#8a8984"
LEVELS = ["n2", "n5", "n10", "n20"]
LEVEL_TICKS = ["n2", "n5\n(default)", "n10", "n20"]
X_TITLE = "Detour / scout aggressiveness"
DODGE = 0.08  # small horizontal offset per family so their error bars stay apart


def gain(data, cond, key, rng):
    """(%, lo, hi): 100 * (none - cond) / none, paired by seed."""
    seeds = sorted(set(data["none"]["runs"]["task_mean"]) & set(data[cond]["runs"]["task_mean"]))
    g = lambda ss: 100 * (_value(data, "none", key, ss) - _value(data, cond, key, ss)) / _value(data, "none", key, ss)  # noqa: E731
    pick = rng.integers(0, len(seeds), size=(BOOT, len(seeds)))
    boots = [g([seeds[i] for i in row]) for row in pick]
    return g(seeds), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def check_ci(data, cond, rng):
    """(mean, lo, hi) check-trip driving in m/h over the seeds."""
    from mattbot_sim.experiment.report import mean_ci

    return mean_ci(list(data[cond]["runs"]["check"].values()), rng)


def plot(panels, stem, rng):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    plt.rcParams.update({"font.size": FONT_SIZE})
    table, paths = [], []
    x = np.arange(len(LEVELS))
    for key, title in METRICS:
        fig, axes = plt.subplots(2, len(panels), figsize=(FIG_W, FIG_H), facecolor=SURFACE, squeeze=False,
                                 sharex=True, sharey="row", gridspec_kw={"height_ratios": [3, 2]})
        for col_i, (name, data) in enumerate(panels):
            top, bottom = axes[0][col_i], axes[1][col_i]
            for ax in (top, bottom):
                ax.set_facecolor(SURFACE)
                ax.grid(True, axis="y", color=GRID, lw=0.8)
                ax.set_axisbelow(True)
            if "opportunistic" in data:
                y, lo, hi = gain(data, "opportunistic", key, rng)
                top.axhline(y, color=REFERENCE, ls=":", lw=1.5, label="Opportunistic")
                table.append([key, name, "Opportunistic", "", "opportunistic", "0.0", "%.2f" % y, "%.2f" % lo,
                              "%.2f" % hi])
            if "oracle" in data and key != "know":  # the oracle's time to know is the whole None value (100 %)
                top.axhline(gain(data, "oracle", key, rng)[0], color=REFERENCE, ls="--", lw=1.2, label="Oracle")
            for k, (family, col, levels) in enumerate(FAMILIES):
                conds = dict(levels)
                keep = [i for i, lab in enumerate(LEVELS) if conds.get(lab) in data]
                xs = x[keep] + (k - (len(FAMILIES) - 1) / 2.0) * DODGE
                g = [gain(data, conds[LEVELS[i]], key, rng) for i in keep]
                d = [check_ci(data, conds[LEVELS[i]], rng) for i in keep]
                for ax, vals, lab in ((top, g, family), (bottom, d, None)):
                    m = [v[0] for v in vals]
                    err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
                    ax.errorbar(xs, m, yerr=err, fmt="-o", color=col, lw=2.0, ms=7, capsize=3, elinewidth=1.2,
                                label=lab)
                table += [[key, name, family, LEVELS[i], conds[LEVELS[i]], "%.1f" % dv[0]] + ["%.2f" % v for v in gv]
                          for i, gv, dv in zip(keep, g, d)]
            top.set_title(map_label(name))
            bottom.set_xticks(x)
            bottom.set_xticklabels(LEVEL_TICKS)
            bottom.set_xlabel(X_TITLE if len(panels) == 1 else None)
            bottom.set_xlim(-0.4, len(LEVELS) - 0.6)
        axes[1][0].set_ylim(bottom=0)  # after every panel is drawn (the row shares its y axis)
        if len(panels) > 1:
            fig.supxlabel(X_TITLE, fontsize=FONT_SIZE)
        axes[0][0].set_ylabel("\n".join(textwrap.wrap("%s: improvement over None (%%)" % title, 26)))
        axes[1][0].set_ylabel("Check-trip\ndriving (m/h)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i] in ("Opportunistic", "Oracle"))
        # Exactly FIG_W x FIG_H: the legend gets a row inside the figure, the panels the rest
        fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=len(labels), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        paths += save(fig, "%s_%s.png" % (stem, key), exact=True)
        plt.close(fig)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "map", "family", "level", "condition", "check_m_per_h", "gain_pct", "lo", "hi"])
        w.writerows(table)
    return paths + [stem + ".csv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="", help="PNG path stem (default: policy.png in the first directory's parent)")
    args = ap.parse_args()
    panels = [(name, data) for name, _attrs, data in (load(d) for d in args.dirs)]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dirs[0])), "policy.png")
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print("wrote\n  " + "\n  ".join(plot(panels, stem, np.random.default_rng(0))))


if __name__ == "__main__":
    main()
