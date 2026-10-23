"""Sweep report: several experiment run directories (e.g. main + fleet1 + fleet5) as levels of one factor.

Each directory's spec gives its level: the map, the number of robots, the fleet's task rate, the number of blocking
obstacles and the obstacles' lifetime range. Per level and condition:
obstacles known and dropped, the Kaplan-Meier median time to drop, ghost-blocked time per run, distance on check
trips per hour, distance per task and (with an oracle condition) the share of the oracle's saving.
"""

import os

import numpy as np

from mattbot_sim.experiment.report import (
    BOOT, _fmt, add_fleet_metrics, add_stale_cost, km_median, mean_ci, restricted_mean_ci, share_ok,
)


def load_level(run_dir):
    import pandas as pd

    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(os.path.join(run_dir, "spec.yaml"))
    runs = pd.read_csv(os.path.join(run_dir, "runs.csv"))
    obs = pd.read_csv(os.path.join(run_dir, "obstacles.csv"))
    tasks_path = os.path.join(run_dir, "tasks.csv")
    runs = add_fleet_metrics(add_stale_cost(runs, obs), pd.read_csv(tasks_path) if os.path.exists(tasks_path) else None)
    level = {"dir": run_dir, "name": spec.name, "map": spec.map, "robots": len(spec.robots),
             "rate_per_min": round(spec.release_rate_per_min, 3),
             "rate_per_robot_per_min": round(spec.release_rate_per_min / max(len(spec.robots), 1), 3),
             "blocking": int(spec.obstacle_types.get("blocking", {}).get("count", 0)),
             "lifetime_s": "%g-%g" % tuple(float(v) for v in spec.obstacles["lifetime_s"])}
    return level, runs, obs


def oracle_share(runs, cond, baseline="none", rng=None):
    """(share, lo, hi) of the oracle's saving in distance per task (paired by seed), or NaNs (also when the oracle's
    own saving is not known to within +-50 %: report.share_ok)."""
    rng = rng or np.random.default_rng(0)
    idx = {c: runs[runs["condition"] == c].set_index("seed")["dist_per_task_m"] for c in (cond, baseline, "oracle")}
    seeds = sorted(set(idx[cond].index) & set(idx[baseline].index) & set(idx["oracle"].index))
    if len(seeds) < 2:
        return float("nan"), float("nan"), float("nan")
    save = np.array([idx[baseline][s] - idx[cond][s] for s in seeds], dtype=float)
    full = np.array([idx[baseline][s] - idx["oracle"][s] for s in seeds], dtype=float)
    if not share_ok(mean_ci(full, rng)):
        return float("nan"), float("nan"), float("nan")
    pick = rng.integers(0, len(seeds), size=(BOOT, len(seeds)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = save[pick].mean(axis=1) / full[pick].mean(axis=1)
    boots = boots[np.isfinite(boots)]
    share = save.mean() / full.mean() if full.mean() else float("nan")
    if len(boots) < 2:
        return share, float("nan"), float("nan")
    return share, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sweep_report(run_dirs, out_dir, factor=None, baseline="none"):
    """Markdown table and a PNG over the levels; factor: "map", "robots", "rate_per_robot_per_min", "blocking",
    "lifetime_s" or None (the one that varies). Returns the markdown."""
    levels = [load_level(d) for d in run_dirs]
    if factor is None:
        factor = next((f for f in ("map", "robots", "rate_per_robot_per_min", "rate_per_min", "blocking", "lifetime_s")
                       if len({lv[f] for lv, _r, _o in levels}) > 1), "robots")
    if factor == "lifetime_s":
        levels.sort(key=lambda t: float(t[0][factor].split("-")[0]))
    elif factor != "map":
        levels.sort(key=lambda t: t[0][factor])  # maps keep the order given
    rng = np.random.default_rng(0)
    conditions = list(dict.fromkeys(c for _lv, runs, _o in levels for c in runs["condition"]))
    lines = ["# Sweep over %s" % factor, "",
             "Levels: %s." % ", ".join("%s (%s = %s)" % (lv["name"], factor, lv[factor]) for lv, _r, _o in levels), "",
             "| %s | condition | known / appeared | dropped / known-removed | KM median (s) | ghost-blocked per run (s) | check trips (m/h) | stale cost (m/h) | net saving vs %s (m/h) | distance per task (m) | saving vs %s (m per task) | share of oracle |" % (factor, baseline, baseline),
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for lv, runs, obs in levels:
        for c in conditions:
            r = runs[runs["condition"] == c]
            if not len(r):
                continue
            oc = obs[obs["condition"] == c]
            app = oc[oc["appeared"] == 1]
            o = oc[oc["latency_s"].notna() | (oc["censored"] == 1)]
            times = np.where(o["censored"] == 1, o["ghost_s"], o["latency_s"]).astype(float)
            cens = (o["censored"] == 1).to_numpy()
            med = km_median(times, cens) if len(o) else float("nan")
            avg = restricted_mean_ci(times, cens, rng) if len(o) else (float("nan"),) * 3
            share = oracle_share(runs, c, baseline, rng) if "oracle" in set(runs["condition"]) and c not in (
                baseline, "oracle") else (float("nan"),) * 3
            ghost = mean_ci(r["ghost_s"], rng)
            check = mean_ci(r["dist_check_m_per_h"], rng) if "dist_check_m_per_h" in r else (float("nan"),) * 3
            table[(lv[factor], c)] = {"km_median": med, "avg_know": avg, "ghost": ghost, "check": check, "share": share}
            b = runs[runs["condition"] == baseline].set_index("seed")["dist_per_task_m"]
            mine = r.set_index("seed")["dist_per_task_m"]
            paired = [b[s] - mine[s] for s in sorted(set(b.index) & set(mine.index))]
            no_share = "-" if c in (baseline, "oracle") or "oracle" not in set(runs["condition"]) else "n/s"
            stale = net = "-"
            if "stale_cost_m_per_h" in r.columns:
                bsc = runs[runs["condition"] == baseline].set_index("seed")
                rr = r.set_index("seed")
                ss = sorted(set(bsc.index) & set(rr.index))
                stale = _fmt(*mean_ci(r["stale_cost_m_per_h"], rng))
                if c != baseline:
                    net = _fmt(*mean_ci([bsc.loc[s, "stale_cost_m_per_h"] - rr.loc[s, "stale_cost_m_per_h"]
                                         - rr.loc[s, "dist_check_m_per_h"] for s in ss], rng))
                table[(lv[factor], c)]["stale"] = mean_ci(r["stale_cost_m_per_h"], rng)
            lines.append("| %s | %s | %d / %d | %d / %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                lv[factor], c, int(app["known"].sum()), len(app), int((~cens).sum()), len(o),
                "-" if np.isnan(med) else ("inf" if np.isinf(med) else "%.0f" % med), _fmt(*ghost, digits=0),
                _fmt(*check), stale, net, _fmt(*mean_ci(r["dist_per_task_m"], rng), digits=2),
                "-" if c == baseline else _fmt(*mean_ci(paired, rng), digits=2),
                no_share if np.isnan(share[0]) else _fmt(*(100 * np.array(share)), digits=0) + " %"))
    lines += ["", "## Fleet summary (per run; mean [95% CI])", "",
              "| %s | condition | time to know: average (s) | time to know: KM median (s) | total distance (m) | distance without scouts (m) | tasks done | total task time (s) | mean task time (s) | mean service (s) |" % factor,
              "|---|---|---|---|---|---|---|---|---|---|"]
    for lv, runs, obs in levels:
        for c in conditions:
            r = runs[runs["condition"] == c]
            if not len(r) or "task_time_total_s" not in r.columns:
                continue
            km = table.get((lv[factor], c), {}).get("km_median", float("nan"))
            avg = table.get((lv[factor], c), {}).get("avg_know", (float("nan"),) * 3)
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                lv[factor], c, _fmt(*avg, digits=0), "-" if np.isnan(km) else ("never" if np.isinf(km) else "%.0f" % km),
                _fmt(*mean_ci(r["dist_total_m"], rng), digits=0), _fmt(*mean_ci(r["dist_no_scout_m"], rng), digits=0),
                _fmt(*mean_ci(r["tasks_done"], rng)), _fmt(*mean_ci(r["task_time_total_s"], rng), digits=0),
                _fmt(*mean_ci(r["task_time_mean_s"], rng), digits=1), _fmt(*mean_ci(r["service_time_mean_s"], rng), digits=1)))
    text = "\n".join(lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sweep_%s.md" % factor), "w") as f:
        f.write(text)
    _plot(table, [lv[factor] for lv, _r, _o in levels], conditions, factor, out_dir, baseline)
    return text


def _plot(table, xs, conditions, factor, out_dir, baseline):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for c in conditions:
        if c == baseline:
            continue
        pts = [(x, table[(x, c)]) for x in xs if (x, c) in table]
        if not pts:
            continue
        x = [p[0] for p in pts]
        axes[0].plot(x, [p[1]["ghost"][0] for p in pts], "o-", label=c)
        med = [p[1]["km_median"] for p in pts]
        axes[1].plot(x, [m if np.isfinite(m) else np.nan for m in med], "o-", label=c)
        if c != "oracle":
            axes[2].plot(x, [p[1]["check"][0] for p in pts], "o-", label=c)
    if any((x, baseline) in table for x in xs):
        axes[0].plot(xs, [table[(x, baseline)]["ghost"][0] if (x, baseline) in table else np.nan for x in xs], "k--",
                     label=baseline)
    for ax, title in zip(axes, ("ghost-blocked time per run (s)", "KM median time to drop (s)",
                                "distance on check trips (m/h)")):
        ax.set_xlabel(factor)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_%s.png" % factor), dpi=110)
    plt.close(fig)
