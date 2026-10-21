"""Run an experiment: every (condition, seed) on the seed's instance, in parallel processes.

Writes to the output directory:
  instances/seed<k>.json, .yaml   the instance and its FastSim scenario (shared by every condition)
  runs.csv                        one row per run: condition, seed, work, distance, checks, obstacle summary
  obstacles.csv                   one row per obstacle per run: when it appeared / was removed / known / dropped
  tasks.csv                       one row per task per run
  spec.yaml                       a copy of the spec
"""

import csv
import json
import math
import multiprocessing
import os
import shutil
import time
from collections import OrderedDict

from mattbot_sim.experiment.generate import DEFAULT_CACHE_DIR, make_instance, setup, write_scenario
from mattbot_sim.experiment.spec import load_spec


def copy_spec(spec, out_dir):
    """spec.yaml and its landmarks file in the run directory (the copy refers to the copied landmarks)."""
    import yaml

    from mattbot_sim.experiment.spec import read_spec_dict

    d = read_spec_dict(spec.path)  # with extends resolved
    name = os.path.basename(spec.landmarks_path)
    shutil.copy(spec.landmarks_path, os.path.join(out_dir, name))
    d["landmarks"] = name
    with open(os.path.join(out_dir, "spec.yaml"), "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def prepare(spec, out_dir, seeds=None, cache_dir=DEFAULT_CACHE_DIR, progress=None):
    """Candidates and one instance per seed. Returns {seed: (instance, scenario_path)}."""
    ctx, landmarks, candidates, conn = setup(spec, cache_dir, progress)
    out = {}
    os.makedirs(os.path.join(out_dir, "instances"), exist_ok=True)
    for seed in seeds if seeds is not None else spec.seeds:
        inst = make_instance(spec, seed, candidates, landmarks, conn)
        base = os.path.join(out_dir, "instances", "seed%d" % seed)
        with open(base + ".json", "w") as f:
            json.dump(inst, f, indent=1)
        out[seed] = (inst, write_scenario(spec, inst, base + ".yaml"))
    return out, ctx, landmarks, candidates


# ---------- One run ----------

def obstacle_rows(sim, instance):
    """Fleet view of every scheduled obstacle (the fast sim has one ledger: any robot's scorer sees all of it)."""
    scorer = next(iter(sim.robots.values())).scorer
    end = sim.now
    rows = []
    for ob in instance["obstacles"]:
        known = [a["t"] for a in scorer.added if a["true_object"] == ob["id"]]
        removed = [r for r in scorer.removals if r["true_object"] == ob["id"] and r["correct"]]
        t_known = min(known) if known else None
        appeared, gone = ob["appear_s"] <= end, ob["remove_s"] <= end
        t_dropped = min(r["t"] for r in removed) if removed else None
        row = OrderedDict(obstacle=ob["id"], type=ob.get("type", "blocking"), cls=ob["class"], x=ob["x"], y=ob["y"],
                          width=ob["width"], I_o=ob["I_o"],
                          appear_s=ob["appear_s"], remove_s=ob["remove_s"], appeared=int(appeared), removed=int(gone),
                          known=int(t_known is not None), known_s=t_known, known_after_s=None,
                          dropped_s=t_dropped, latency_s=None, censored=0, ghost_s=0.0, dropped_by=None,
                          dropped_by_robot=None)
        if t_known is not None:
            row["known_after_s"] = round(t_known - ob["appear_s"], 1)
        if gone and t_known is not None and t_known <= ob["remove_s"] + 1.0:
            # Known while it was there: the fleet kept avoiding it until the ledger dropped it
            if t_dropped is not None:
                row["latency_s"] = round(t_dropped - ob["remove_s"], 1)
                row["ghost_s"] = row["latency_s"]
                src = [e for e in sim.removal_log
                       if e[3] == ob["class"] and math.hypot(e[4] - ob["x"], e[5] - ob["y"]) < 1.5
                       and e[0] <= t_dropped + 1e-6]
                if src:
                    t, rid, kind, *_rest = max(src, key=lambda e: e[0])
                    row["dropped_by"], row["dropped_by_robot"] = kind, rid
            else:
                row["censored"] = 1  # still in the ledger at the end of the run
                row["ghost_s"] = round(end - ob["remove_s"], 1)
        rows.append(row)
    return rows


def run_row(sim, orchestrator, obstacles, condition, seed, wall_s):
    summaries = sim.summaries()
    tasks = orchestrator.tasks
    done = [t for t in tasks if t["status"] == "done"]
    row = OrderedDict(condition=condition, seed=seed, run_s=round(sim.now, 1), wall_s=round(wall_s, 1),
                      tasks=len(tasks), tasks_done=len(done), tasks_failed=sum(t["status"] == "failed" for t in tasks),
                      tasks_unfinished=sum(t["status"] not in ("done", "failed") for t in tasks),
                      robot_stuck_events=getattr(orchestrator, "stuck_events", 0),
                      makespan_s=max((t["done_s"] for t in done), default=None),
                      mean_task_s=round(sum(t["done_s"] - t["release_s"] for t in done) / len(done), 1) if done else None)
    dist = OrderedDict()
    for s in summaries.values():
        for k, v in s["navigator"].items():
            if k.startswith(("dist_", "t_", "time_")) or k in ("replans", "plan_failures"):
                dist[k] = dist.get(k, 0.0) + float(v)
    for k in ("dist_total_m", "dist_task_m", "dist_detour_m", "dist_scout_m", "dist_return_m", "dist_other_m",
              "replans", "plan_failures"):
        row[k] = round(dist.get(k, 0.0), 1)
    row["dist_per_task_m"] = round(row["dist_total_m"] / len(done), 2) if done else None
    # Cost of checking: driving for detours / scouts (and back), and how the robots' time was spent
    row["dist_check_m"] = round(row["dist_detour_m"] + row["dist_scout_m"] + row["dist_return_m"], 1)
    hours = sim.now / 3600.0
    row["dist_check_m_per_h"] = round(row["dist_check_m"] / hours, 1) if hours > 0 else None
    robot_s = sim.now * len(summaries)
    if robot_s > 0:
        check_s = sum(dist.get("t_%s_s" % m, 0.0) for m in ("OBSERVE_TURN", "OBSERVE_DWELL"))
        trips_s = sum(dist.get("time_%s_s" % a, 0.0) for a in ("detour", "scout", "return"))
        task_s = dist.get("time_task_s", 0.0)
        row["share_looking"] = round(check_s / robot_s, 4)  # turning to look and dwelling, on any trip
        row["share_check_trips"] = round(trips_s / robot_s, 4)  # out on detours / scouts and back
        row["share_task"] = round(task_s / robot_s, 4)  # with a task goal (includes looking on the way)
        row["share_idle"] = round(max(robot_s - task_s - trips_s, 0.0) / robot_s, 4)
    for kind in ("OPPORTUNISTIC", "AT_START", "AT_GOAL", "DETOUR", "SCOUT"):
        row["checks_%s" % kind.lower()] = sum(s["observation_stops_by_kind"][kind]["ENDED"] for s in summaries.values())
    for key in ("detours_started", "detours_yielded", "scouts_started", "scouts_done", "scouts_yielded",
                "scouts_cancelled", "stops_skipped"):
        row[key] = sum(s.get(key, 0) for s in summaries.values())
    first = next(iter(summaries.values()))  # one fleet ledger: the same for every robot
    row["false_removals"] = first["false_removals"]
    row["obstacles_known"] = sum(o["known"] for o in obstacles)
    removed_known = [o for o in obstacles if o["latency_s"] is not None or o["censored"]]
    row["obstacles_removed_known"] = len(removed_known)
    row["obstacles_dropped"] = sum(o["latency_s"] is not None for o in obstacles)
    row["obstacles_censored"] = sum(o["censored"] for o in obstacles)
    lat = [o["latency_s"] for o in obstacles if o["latency_s"] is not None]
    row["latency_mean_s"] = round(sum(lat) / len(lat), 1) if lat else None
    row["ghost_s"] = round(sum(o["ghost_s"] for o in obstacles), 1)
    return row


def run_one(job):
    """One (condition, seed) run (top level, so multiprocessing can pickle it)."""
    spec_path, condition, seed, scenario_path, instance, cache_dir = job
    from mattbot_sim.experiment.orchestrator import orchestrator_for
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    spec = load_spec(spec_path)
    settings = spec.params_of(condition)
    params = FastParams().update(settings)
    params._explicit = dict(settings)
    orch = orchestrator_for(spec, instance)
    t0 = time.time()  # wall clock: run time
    sim = FastSim(scenario_path, params=params, seed=seed, orchestrator=orch, duration=spec.max_duration_s,
                  roadmap_cache_dir=cache_dir)
    sim.run()
    obstacles = obstacle_rows(sim, instance)
    row = run_row(sim, orch, obstacles, condition, seed, time.time() - t0)  # wall clock: run time
    tasks = [OrderedDict(condition=condition, seed=seed, task=t["id"], landmark=t["landmark"], release_s=t["release_s"],
                         assigned_s=t["assigned_s"], done_s=t["done_s"], status=t["status"], robot=t["robot"],
                         retries=t["retries"], handoffs=t.get("handoffs", 0)) for t in orch.tasks]
    for o in obstacles:
        o["condition"], o["seed"] = condition, seed
    return row, obstacles, tasks


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(spec_path, out_dir, conditions=None, seeds=None, jobs=1, cache_dir=DEFAULT_CACHE_DIR, progress=None):
    spec = load_spec(spec_path)
    os.makedirs(out_dir, exist_ok=True)
    copy_spec(spec, out_dir)
    seeds = spec.seeds if seeds is None else seeds
    conditions = list(spec.conditions) if conditions is None else conditions
    instances, *_ = prepare(spec, out_dir, seeds, cache_dir)
    work = [(spec.path, c, s, instances[s][1], instances[s][0], cache_dir) for s in seeds for c in conditions]
    results = []
    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for k, res in enumerate(pool.imap_unordered(run_one, work)):
                results.append(res)
                if progress:
                    progress(k + 1, len(work), res[0])
    else:
        for k, job in enumerate(work):
            results.append(run_one(job))
            if progress:
                progress(k + 1, len(work), results[-1][0])
    order = {c: i for i, c in enumerate(conditions)}
    results.sort(key=lambda r: (r[0]["seed"], order[r[0]["condition"]]))
    write_csv(os.path.join(out_dir, "runs.csv"), [r[0] for r in results])
    write_csv(os.path.join(out_dir, "obstacles.csv"), [o for r in results for o in r[1]])
    write_csv(os.path.join(out_dir, "tasks.csv"), [t for r in results for t in r[2]])
    return results
