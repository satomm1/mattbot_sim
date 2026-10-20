"""Report of an experiment run directory (runs.csv, obstacles.csv): tables, paired differences and plots.

Experiment 1 (time to learn an obstacle is gone): per condition, over the obstacles that were known while there
and later removed, the latency from the true removal to the ledger drop. Obstacles still in the ledger at the end
of a run are censored there; the median and the curves use Kaplan-Meier, and the fraction censored is reported.
Experiment 2 (distance): fleet distance per run (and per task), and its paired difference to the baseline
condition on the same seed, with 95 % bootstrap confidence intervals.
"""

import os

import numpy as np

BOOT = 2000


def mean_ci(values, rng=None, n=BOOT):
    """(mean, lo, hi) with a 95 % bootstrap interval; NaN for no data."""
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    boots = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def kaplan_meier(times, censored):
    """Survival curve of "still in the ledger": ([t], [S(t)]) at each drop time."""
    t = np.asarray(times, dtype=float)
    c = np.asarray(censored, dtype=bool)
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    at_risk, s = len(t), 1.0
    xs, ys = [0.0], [1.0]
    for ti in np.unique(t):
        here = t == ti
        events = int((here & ~c).sum())
        if events and at_risk:
            s *= 1.0 - events / at_risk
            xs.append(float(ti))
            ys.append(s)
        at_risk -= int(here.sum())
    return xs, ys


def km_median(times, censored):
    xs, ys = kaplan_meier(times, censored)
    for x, y in zip(xs, ys):
        if y <= 0.5:
            return x
    return float("inf")  # more than half never dropped


def _fmt(m, lo, hi, digits=1):
    if np.isnan(m):
        return "-"
    if np.isnan(lo):
        return ("%." + str(digits) + "f") % m
    return ("%." + str(digits) + "f [%." + str(digits) + "f, %." + str(digits) + "f]") % (m, lo, hi)


def report(run_dir, baseline=None, latency_baseline=None):
    import pandas as pd

    runs = pd.read_csv(os.path.join(run_dir, "runs.csv"))
    obs = pd.read_csv(os.path.join(run_dir, "obstacles.csv"))
    conditions = list(dict.fromkeys(runs["condition"]))
    baseline = baseline or conditions[0]
    latency_baseline = latency_baseline or next((c for c in conditions if c != baseline), baseline)
    rng = np.random.default_rng(0)
    lines = ["# Experiment report: %s" % os.path.basename(os.path.abspath(run_dir)), "",
             "%d runs, conditions %s, seeds %s." % (len(runs), ", ".join(conditions),
                                                    ", ".join(str(s) for s in sorted(set(runs["seed"])))), ""]

    # ---- Work and distance ----
    lines += ["## Work and distance (fleet, per run; mean [95% CI])", "",
              "| condition | tasks done | failed | makespan (s) | distance (m) | per task (m) | detour (m) | scout+return (m) | checks (opp / end / detour / scout) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in conditions:
        r = runs[runs["condition"] == c]
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %.1f / %.1f / %.1f / %.1f |" % (
            c, _fmt(*mean_ci(r["tasks_done"], rng)), int(r["tasks_failed"].sum()),
            _fmt(*mean_ci(r["makespan_s"], rng), digits=0), _fmt(*mean_ci(r["dist_total_m"], rng)),
            _fmt(*mean_ci(r["dist_per_task_m"], rng), digits=2), _fmt(*mean_ci(r["dist_detour_m"], rng)),
            _fmt(*mean_ci(r["dist_scout_m"] + r["dist_return_m"], rng)),
            r["checks_opportunistic"].mean(), (r["checks_at_start"] + r["checks_at_goal"]).mean(),
            r["checks_detour"].mean(), r["checks_scout"].mean()))
    base = runs[runs["condition"] == baseline].set_index("seed")
    lines += ["", "Paired difference to `%s` on the same seed (negative = less driving):" % baseline, "",
              "| condition | distance (m) | distance (%) | per task (m) | makespan (s) | seeds |", "|---|---|---|---|---|---|"]
    for c in conditions:
        if c == baseline:
            continue
        r = runs[runs["condition"] == c].set_index("seed")
        seeds = sorted(set(r.index) & set(base.index))
        d = [r.loc[s, "dist_total_m"] - base.loc[s, "dist_total_m"] for s in seeds]
        pct = [100.0 * x / base.loc[s, "dist_total_m"] for x, s in zip(d, seeds) if base.loc[s, "dist_total_m"]]
        dt = [r.loc[s, "dist_per_task_m"] - base.loc[s, "dist_per_task_m"] for s in seeds]
        dm = [r.loc[s, "makespan_s"] - base.loc[s, "makespan_s"] for s in seeds]
        lines.append("| %s | %s | %s | %s | %s | %d |" % (c, _fmt(*mean_ci(d, rng)), _fmt(*mean_ci(pct, rng)),
                                                       _fmt(*mean_ci(dt, rng), digits=2), _fmt(*mean_ci(dm, rng), digits=0),
                                                       len(seeds)))

    # ---- Obstacles ----
    lines += ["", "## Time to learn an obstacle is gone", "",
              "Obstacles known while there and removed during the run; censored = still in the ledger at the end.", "",
              "| condition | obstacles | dropped | censored | latency mean of dropped (s) | KM median (s) | ghost-blocked per run (s) | dropped by |",
              "|---|---|---|---|---|---|---|---|"]
    curves = {}
    for c in conditions:
        o = obs[(obs["condition"] == c) & (obs["latency_s"].notna() | (obs["censored"] == 1))]
        # A censored obstacle was still in the ledger at the end of the run, ghost_s after its removal
        times = np.where(o["censored"] == 1, o["ghost_s"], o["latency_s"]).astype(float)
        cens = (o["censored"] == 1).to_numpy()
        curves[c] = kaplan_meier(times, cens) if len(o) else ([0.0], [1.0])
        by = o["dropped_by"].dropna().value_counts()
        lines.append("| %s | %d | %d | %d (%.0f%%) | %s | %s | %s | %s |" % (
            c, len(o), int((~cens).sum()), int(cens.sum()), 100.0 * cens.mean() if len(o) else 0.0,
            _fmt(*mean_ci(o["latency_s"].dropna(), rng)),
            ("%.0f" % km_median(times, cens)) if len(o) else "-",
            _fmt(*mean_ci(runs[runs["condition"] == c]["ghost_s"], rng), digits=0),
            ", ".join("%s %d" % (k, v) for k, v in by.items()) or "-"))
    known = obs[obs["appeared"] == 1]
    lines += ["", "Obstacles that appeared: %d per condition on average; known to the fleet: %.0f%%." % (
        len(known) / max(len(conditions), 1), 100.0 * known["known"].mean() if len(known) else 0.0)]

    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(text)
    _plots(run_dir, runs, conditions, baseline, curves)
    return text


def _plots(run_dir, runs, conditions, baseline, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for c, (xs, ys) in curves.items():
        ax.step(xs, ys, where="post", label=c)
    ax.set_xlabel("time since removal (s)")
    ax.set_ylabel("fraction still in the ledger (Kaplan-Meier)")
    ax.set_title("Time to learn an obstacle is gone")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    data = [runs[runs["condition"] == c]["dist_total_m"].dropna() for c in conditions]
    ax.boxplot(data, labels=conditions, showmeans=True)
    ax.set_ylabel("fleet distance per run (m)")
    ax.set_title("Distance to complete the tasks")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    base = runs[runs["condition"] == baseline].set_index("seed")["dist_total_m"]
    others = [c for c in conditions if c != baseline]
    diffs = []
    for c in others:
        r = runs[runs["condition"] == c].set_index("seed")["dist_total_m"]
        seeds = sorted(set(r.index) & set(base.index))
        diffs.append([r[s] - base[s] for s in seeds])
    if others:
        ax.boxplot(diffs, labels=others, showmeans=True)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("distance - %s on the same seed (m)" % baseline)
    ax.set_title("Paired difference")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "report.png"), dpi=110)
    plt.close(fig)
