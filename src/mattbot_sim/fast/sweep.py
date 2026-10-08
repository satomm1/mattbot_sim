"""Parameter sweeps: every combination of a parameter grid x seeds, in parallel processes, as CSV rows.

grid: {"observe_cooldown_s": [30, 60, 120], ...} (FastParams names). One row per (run, robot): the parameters,
seed, robot_id and the flattened summary (nested fields as dotted names, lists as their length).
aggregate() groups rows by parameter set and gives mean / std of every numeric column.
"""

import csv
import itertools
import math
import multiprocessing
import os
from collections import OrderedDict


def flatten(d, prefix=""):
    out = OrderedDict()
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = len(v)
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)) or v is None:
            out[key] = v
        else:
            out[key] = str(v)
    return out


def combinations(grid, seeds):
    names = sorted(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        for seed in seeds:
            yield dict(zip(names, values)), seed


def run_one(job):
    """One run (top level, so multiprocessing can pickle it). job = (scenario, settings, seed, duration, cache)."""
    scenario, settings, seed, duration, cache_dir = job
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    params = FastParams().update(settings)
    params._explicit = dict(settings)
    sim = FastSim(scenario, params=params, seed=seed, duration=duration, roadmap_cache_dir=cache_dir)
    rows = []
    for rid, summary in sim.run().items():
        row = OrderedDict(settings)
        row["seed"] = seed
        row["robot_id"] = rid
        row.update(flatten({k: v for k, v in summary.items() if k != "expectations"}))
        rows.append(row)
    return rows


def sweep(scenario, grid, seeds=(0,), jobs=1, duration=None, cache_dir=None, progress=None):
    work = [(scenario, settings, seed, duration, cache_dir) for settings, seed in combinations(grid, seeds)]
    rows = []
    if jobs <= 1:
        for k, job in enumerate(work):
            rows.extend(run_one(job))
            if progress:
                progress(k + 1, len(work))
    else:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for k, result in enumerate(pool.imap_unordered(run_one, work)):
                rows.extend(result)
                if progress:
                    progress(k + 1, len(work))
    return rows


def aggregate(rows, grid_names):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(tuple(row.get(n) for n in grid_names) + (row.get("robot_id"),), []).append(row)
    out = []
    for key, members in groups.items():
        agg = OrderedDict(zip(list(grid_names) + ["robot_id"], key))
        agg["runs"] = len(members)
        columns = [c for c in members[0] if c not in agg and c != "seed"]
        for c in columns:
            vals = [m.get(c) for m in members if isinstance(m.get(c), (int, float)) and not isinstance(m.get(c), bool)]
            if len(vals) == len(members) and vals:
                mean = sum(vals) / len(vals)
                agg[c + ".mean"] = round(mean, 4)
                agg[c + ".std"] = round(math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)), 4)
        out.append(agg)
    return out


def write_csv(path, rows):
    if not rows:
        return
    columns = list(OrderedDict((c, None) for row in rows for c in row))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
