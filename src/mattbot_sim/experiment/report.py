"""Report of an experiment run directory (runs.csv, obstacles.csv): tables, paired differences and plots.

Experiment 1 (time to learn an obstacle is gone): per condition, over the obstacles that were known while there
and later removed, the latency from the true removal to the ledger drop. Obstacles still in the ledger at the end
of a run are censored there; the median and the curves use Kaplan-Meier, and the fraction censored is reported.
Experiment 2 (distance): fleet distance per run (and per task), and its paired difference to the baseline
condition on the same seed, with 95 % bootstrap confidence intervals.
Per obstacle type (spec obstacles.types), the latency table is repeated by type. The cost of checking: distance
driven for detours / scouts per hour, mean task time, and how the robots' time was spent. With an `oracle`
condition (the ledger drops an obstacle the moment it is removed), each condition's saving in distance per task
over the baseline is also given as a share of the oracle's.
Stale-obstacle cost (importance-weighted): the extra driving stale obstacles cause, per run and hour,
    sum over obstacles of ghost_s x I_o x trip rate (completed tasks per second of the run),
i.e. what the robots' value model trades against the driving spent on checks. Obstacles that block nothing (I_o = 0:
on_route, most visible ones) cost nothing here, however long they stay in the ledger; ghost-blocked time still counts
them (ledger accuracy). The trade-off plot shows this saving over the baseline against the check-trip driving, both in
m per hour, with the break-even line.
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


def add_fleet_metrics(runs, tasks=None):
    """Per run: total distance with and without scout driving (out to scout viewpoints and back; detours count as
    the robot's own trip), and task times from tasks.csv: total and mean release -> arrival over completed tasks, and
    its split into waiting for a robot (release -> assigned) and service (assigned -> arrival)."""
    runs = runs.copy()
    if "dist_total_m" in runs.columns:
        scout = runs.get("dist_scout_m", 0.0) + runs.get("dist_return_m", 0.0)
        runs["dist_no_scout_m"] = runs["dist_total_m"] - scout
    if tasks is not None and len(tasks):
        done = tasks[tasks["status"] == "done"].copy()
        done["task_s"] = done["done_s"] - done["release_s"]
        done["wait_s"] = done["assigned_s"] - done["release_s"]
        done["service_s"] = done["done_s"] - done["assigned_s"]
        g = done.groupby(["condition", "seed"])
        stats = {"task_time_total_s": g["task_s"].sum(), "task_time_mean_s": g["task_s"].mean(),
                 "wait_time_mean_s": g["wait_s"].mean(), "service_time_mean_s": g["service_s"].mean()}
        for col, ser in stats.items():
            runs[col] = [float(ser.get((c, s), float("nan"))) for c, s in zip(runs["condition"], runs["seed"])]
    return runs


def add_stale_cost(runs, obs):
    """runs["stale_cost_m_per_h"]: sum of ghost_s x I_o x trip rate over the run's obstacles, per hour (see the module
    docstring). Unchanged if the columns are missing."""
    if not {"ghost_s", "I_o"} <= set(obs.columns) or "run_s" not in runs.columns:
        return runs
    weighted = (obs["ghost_s"].fillna(0.0) * obs["I_o"].fillna(0.0)).groupby([obs["condition"], obs["seed"]]).sum()
    out = []
    for _i, r in runs.iterrows():
        run_s = float(r["run_s"]) or float("nan")
        rate = float(r["tasks_done"]) / run_s  # trips per second
        out.append(float(weighted.get((r["condition"], r["seed"]), 0.0)) * rate / (run_s / 3600.0))
    runs = runs.copy()
    runs["stale_cost_m_per_h"] = out
    return runs


def share_ok(saving_ci):
    """Whether a share of a saving (mean, lo, hi) is meaningful: the saving is known to within +-50 % (lo > mean / 2).
    A share of a small, noisy saving (a map where every blockage is cheap) is a ratio of noise: distance per task
    also varies between conditions because small timing differences change which robot gets which task."""
    m, lo, _hi = saving_ci
    return bool(np.isfinite(lo) and m > 0 and lo > 0.5 * m)


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


RMST_CAP_S = 1800.0  # restricted mean time to know: obstacles still in the ledger are counted up to this


def restricted_mean(times, censored, cap=RMST_CAP_S):
    """Restricted mean time (area under the Kaplan-Meier curve up to cap): the average time to know an obstacle is
    gone, counting each obstacle's full time (an obstacle still in the ledger at the end counts at least its time
    so far; all times are capped at cap). Unlike the median, every obstacle's change counts by its size."""
    if len(times) == 0:
        return float("nan")
    xs, ys = kaplan_meier(times, censored)
    xs, ys = list(xs) + [cap], list(ys) + [ys[-1]]
    return float(sum(ys[i] * max(min(xs[i + 1], cap) - min(xs[i], cap), 0.0) for i in range(len(xs) - 1)))


def restricted_mean_ci(times, censored, rng=None, n=500, cap=RMST_CAP_S):
    """(restricted mean, lo, hi) with a 95 % bootstrap over obstacles."""
    t = np.asarray(times, dtype=float)
    c = np.asarray(censored, dtype=bool)
    if len(t) < 2:
        return restricted_mean(t, c, cap), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(t), len(t))
        boots.append(restricted_mean(t[i], c[i], cap))
    return restricted_mean(t, c, cap), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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
    tasks_path = os.path.join(run_dir, "tasks.csv")
    runs = add_fleet_metrics(add_stale_cost(runs, obs), pd.read_csv(tasks_path) if os.path.exists(tasks_path) else None)
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

    if "type" in obs.columns and obs["type"].nunique() > 1:
        lines += ["", "### By obstacle type", "",
                  "| type | condition | known / appeared | dropped | censored | latency mean of dropped (s) | KM median (s) | dropped by |",
                  "|---|---|---|---|---|---|---|---|"]
        for typ in [t for t in ("blocking", "on_route", "visible") if t in set(obs["type"])]:
            for c in conditions:
                oc = obs[(obs["condition"] == c) & (obs["type"] == typ)]
                o = oc[oc["latency_s"].notna() | (oc["censored"] == 1)]
                times = np.where(o["censored"] == 1, o["ghost_s"], o["latency_s"]).astype(float)
                cens = (o["censored"] == 1).to_numpy()
                by = o["dropped_by"].dropna().value_counts()
                app = oc[oc["appeared"] == 1]
                lines.append("| %s | %s | %d / %d | %d | %d (%.0f%%) | %s | %s | %s |" % (
                    typ, c, int(app["known"].sum()), len(app), int((~cens).sum()), int(cens.sum()),
                    100.0 * cens.mean() if len(o) else 0.0, _fmt(*mean_ci(o["latency_s"].dropna(), rng)),
                    ("%.0f" % km_median(times, cens)) if len(o) else "-",
                    ", ".join("%s %d" % (k, v) for k, v in by.items()) or "-"))

    # ---- Fleet summary: time to know, distance, task time ----
    if "task_time_total_s" in runs.columns:
        lines += ["", "## Fleet summary (per run; mean [95% CI])", "",
                  "Time to know an obstacle is gone: from its removal until the ledger drops it. Average: restricted mean "
                  "(area under the Kaplan-Meier curve up to 1800 s, 95%% bootstrap over obstacles): every obstacle's time "
                  "counts, one still in the ledger at the end at least its time so far; the KM median is given for "
                  "reference (it only counts obstacles on either side of the middle). Distance: the whole fleet; without "
                  "scouts leaves out "
                  "the driving out to scout viewpoints and back. Task time: release -> arrival, summed and averaged over "
                  "completed tasks, split into waiting for a robot and service (assigned -> arrival).", "",
                  "| condition | time to know: average (s) | time to know: KM median (s) | total distance (m) | distance without scouts (m) | tasks done | total task time (s) | mean task time (s) | mean wait (s) | mean service (s) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in conditions:
            r = runs[runs["condition"] == c]
            o = obs[(obs["condition"] == c) & (obs["latency_s"].notna() | (obs["censored"] == 1))]
            times = np.where(o["censored"] == 1, o["ghost_s"], o["latency_s"]).astype(float)
            cens = (o["censored"] == 1).to_numpy()
            med = km_median(times, cens) if len(o) else float("nan")
            avg = restricted_mean_ci(times, cens, rng) if len(o) else (float("nan"),) * 3
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                c, _fmt(*avg, digits=0), "-" if np.isnan(med) else ("never" if np.isinf(med) else "%.0f" % med),
                _fmt(*mean_ci(r["dist_total_m"], rng), digits=0),
                _fmt(*mean_ci(r["dist_no_scout_m"], rng), digits=0), _fmt(*mean_ci(r["tasks_done"], rng)),
                _fmt(*mean_ci(r["task_time_total_s"], rng), digits=0), _fmt(*mean_ci(r["task_time_mean_s"], rng), digits=1),
                _fmt(*mean_ci(r["wait_time_mean_s"], rng), digits=1), _fmt(*mean_ci(r["service_time_mean_s"], rng), digits=1)))
        bsum = runs[runs["condition"] == baseline].set_index("seed")
        lines += ["", "Paired difference to `%s` on the same seed (negative = less):" % baseline, "",
                  "| condition | total distance (m) | distance without scouts (m) | total task time (s) | mean task time (s) | mean service (s) |",
                  "|---|---|---|---|---|---|"]
        for c in conditions:
            if c == baseline:
                continue
            r = runs[runs["condition"] == c].set_index("seed")
            ss = sorted(set(r.index) & set(bsum.index))
            d = lambda col: _fmt(*mean_ci([r.loc[s, col] - bsum.loc[s, col] for s in ss], rng),
                                 digits=0 if "total" in col or "dist" in col else 1)
            lines.append("| %s | %s | %s | %s | %s | %s |" % (c, d("dist_total_m"), d("dist_no_scout_m"),
                                                             d("task_time_total_s"), d("task_time_mean_s"),
                                                             d("service_time_mean_s")))

    # ---- Stale-obstacle cost ----
    if "stale_cost_m_per_h" in runs.columns and "dist_check_m_per_h" in runs.columns:
        lines += ["", "## Stale-obstacle cost (importance-weighted; m of extra driving per hour, fleet)", "",
                  "ghost_s x I_o x trip rate summed over the run's obstacles (obstacles that block nothing cost nothing); "
                  "saved and net are paired by seed (net = saved - check trips; > 0: checking pays for itself).", "",
                  "| condition | stale cost (m/h) | saved vs %s (m/h) | check trips (m/h) | net saving (m/h) |" % baseline,
                  "|---|---|---|---|---|"]
        bs = runs[runs["condition"] == baseline].set_index("seed")
        for c in conditions:
            r = runs[runs["condition"] == c].set_index("seed")
            seeds = sorted(set(r.index) & set(bs.index))
            saved = [bs.loc[s, "stale_cost_m_per_h"] - r.loc[s, "stale_cost_m_per_h"] for s in seeds]
            net = [sv - r.loc[s, "dist_check_m_per_h"] for sv, s in zip(saved, seeds)]
            lines.append("| %s | %s | %s | %s | %s |" % (
                c, _fmt(*mean_ci(r["stale_cost_m_per_h"], rng)), "-" if c == baseline else _fmt(*mean_ci(saved, rng)),
                _fmt(*mean_ci(r["dist_check_m_per_h"], rng)), "-" if c == baseline else _fmt(*mean_ci(net, rng))))

    # ---- Cost ----
    if "dist_check_m_per_h" in runs.columns:
        lines += ["", "## Cost of checking (fleet, per run; mean [95% CI])", "",
                  "| condition | check trips (m/h) | mean task time (s) | unfinished tasks | plan failures | detours abandoned (too long) | looking | on check trips | on tasks | idle |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in conditions:
            r = runs[runs["condition"] == c]
            ab = "%d (%d)" % (r["detours_abandoned"].sum(), r["detours_too_long"].sum()) \
                if "detours_too_long" in r.columns else "-"
            lines.append("| %s | %s | %s | %.1f | %.1f | %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
                c, _fmt(*mean_ci(r["dist_check_m_per_h"], rng)), _fmt(*mean_ci(r["mean_task_s"], rng), digits=0),
                r["tasks_unfinished"].mean(), r["plan_failures"].mean(), ab, 100 * r["share_looking"].mean(),
                100 * r["share_check_trips"].mean(), 100 * r["share_task"].mean(), 100 * r["share_idle"].mean()))
        lines += ["", "Time shares are of robot time (run length x robots); \"looking\" (turning to look and dwelling) "
                  "is also part of the task or check trip it happens on."]

    # ---- Oracle share ----
    if "oracle" in conditions and baseline != "oracle":
        orc = runs[runs["condition"] == "oracle"].set_index("seed")
        lines += ["", "## Share of the oracle's saving (distance per task, paired by seed)", "",
                  "(%s - condition) / (%s - oracle), from the mean paired differences; 95%% bootstrap over seeds. "
                  "n/s: the oracle's own saving is not known to within +-50 %% (its CI's lower end is below half its mean), "
                  "so a share would be a ratio of noise." % (
                      baseline, baseline), "",
                  "| condition | saving (m per task) | oracle saving (m per task) | share of oracle |", "|---|---|---|---|"]
        for c in conditions:
            if c in (baseline, "oracle"):
                continue
            r = runs[runs["condition"] == c].set_index("seed")
            seeds = sorted(set(r.index) & set(base.index) & set(orc.index))
            if not seeds:
                continue
            save = np.array([base.loc[s, "dist_per_task_m"] - r.loc[s, "dist_per_task_m"] for s in seeds], dtype=float)
            full = np.array([base.loc[s, "dist_per_task_m"] - orc.loc[s, "dist_per_task_m"] for s in seeds], dtype=float)
            share = save.mean() / full.mean() if full.mean() else float("nan")
            idx = rng.integers(0, len(seeds), size=(BOOT, len(seeds)))
            with np.errstate(divide="ignore", invalid="ignore"):
                boots = save[idx].mean(axis=1) / full[idx].mean(axis=1)
            boots = boots[np.isfinite(boots)]
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 1 else (float("nan"), float("nan")))
            full_ci = mean_ci(full, rng)
            share_txt = (_fmt(100 * share, 100 * lo, 100 * hi, digits=0) + " %") if share_ok(full_ci) else "n/s"
            lines.append("| %s | %s | %s | %s |" % (c, _fmt(*mean_ci(save, rng), digits=2), _fmt(*full_ci, digits=2),
                                                   share_txt))

    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(text)
    _plots(run_dir, runs, conditions, baseline, curves)
    return text


def tradeoff_points(runs, conditions, baseline, rng=None):
    """Per condition: (check trips m/h mean, lo, hi), (stale-obstacle cost saved over baseline, m/h, lo, hi), or the
    ghost-blocked time saved (s per run) without stale_cost_m_per_h."""
    rng = rng or np.random.default_rng(0)
    base = runs[runs["condition"] == baseline].set_index("seed")
    out = {}
    for c in conditions:
        if c == baseline:
            continue
        r = runs[runs["condition"] == c].set_index("seed")
        seeds = sorted(set(r.index) & set(base.index))
        col = "stale_cost_m_per_h" if "stale_cost_m_per_h" in r.columns else "ghost_s"
        saved = [base.loc[s, col] - r.loc[s, col] for s in seeds]
        cost = r["dist_check_m_per_h"] if "dist_check_m_per_h" in r.columns else r["dist_detour_m"] * 0
        out[c] = (mean_ci(cost, rng), mean_ci(saved, rng))
    return out


def _plots(run_dir, runs, conditions, baseline, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for c, (xs, ys) in curves.items():
        ax.step(xs, ys, where="post", label=c)
    ax.set_xlabel("time since removal (s)")
    ax.set_ylabel("fraction still in the ledger (Kaplan-Meier)")
    ax.set_title("Time to learn an obstacle is gone")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for c, ((cm, clo, chi), (sm, slo, shi)) in tradeoff_points(runs, conditions, baseline).items():
        xerr = None if np.isnan(clo) else [[cm - clo], [chi - cm]]
        yerr = None if np.isnan(slo) else [[sm - slo], [shi - sm]]
        ax.errorbar([cm], [sm], xerr=xerr, yerr=yerr, fmt="o", capsize=4, label=c)
        ax.annotate(c, (cm, sm), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("distance on check trips (m per hour, fleet)")
    if "stale_cost_m_per_h" in runs.columns:
        lim = max([1.0] + [v for c in ax.get_lines() for v in list(c.get_xdata()) + list(c.get_ydata())
                          if np.isfinite(v)])
        ax.plot([0, lim], [0, lim], "k:", lw=0.8, label="break-even")
        ax.set_ylabel("stale-obstacle driving saved vs %s (m per hour)" % baseline)
    else:
        ax.set_ylabel("ghost-blocked time saved vs %s (s per run)" % baseline)
    ax.set_title("Benefit vs cost (mean, 95% CI; above the dotted line: pays for itself)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    col = "dist_per_task_m"
    base = runs[runs["condition"] == baseline].set_index("seed")[col]
    others = [c for c in conditions if c != baseline]
    diffs = []
    for c in others:
        r = runs[runs["condition"] == c].set_index("seed")[col]
        seeds = sorted(set(r.index) & set(base.index))
        diffs.append([r[s] - base[s] for s in seeds if not (np.isnan(r[s]) or np.isnan(base[s]))])
    if others:
        ax.boxplot(diffs, labels=others, showmeans=True)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("distance per task - %s on the same seed (m)" % baseline)
    ax.set_title("Paired difference in distance per task")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    if "share_task" in runs.columns:
        parts = [("share_task", "on tasks"), ("share_check_trips", "on check trips"), ("share_idle", "idle")]
        bottom = np.zeros(len(conditions))
        for key, label in parts:
            vals = np.array([runs[runs["condition"] == c][key].mean() * 100 for c in conditions])
            ax.bar(conditions, vals, bottom=bottom, label=label)
            bottom += vals
        look = [runs[runs["condition"] == c]["share_looking"].mean() * 100 for c in conditions]
        ax.plot(conditions, look, "k^", label="looking (part of the above)")
        ax.set_ylabel("share of robot time (%)")
        ax.set_title("How the robots spent their time")
        ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "report.png"), dpi=110)
    plt.close(fig)
