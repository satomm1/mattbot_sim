#!/usr/bin/env python3
"""Run a fleet scenario: several simulated robots on this machine, one ROS master each, talking over DDS.

  rosrun mattbot_sim run_fleet.py fleet_share_removal [--speed 1] [--port-base 11450] [-- observe_detour:=true ...]

For each robot in the scenario's ``robots`` list this starts ``roslaunch mattbot_sim sim.launch`` on its own
master (port-base + k) with ROBOT_ID = the robot's id, dds:=true (loopback-only CycloneDDS, see
config/cyclonedds_sim.xml) and the same clock_epoch, so all robots share one simulated timeline and their
separate worlds see the same objects and events at the same times (the robots do not see or block each
other). Arguments after ``--`` go to every robot's sim.launch. Logs and results are written to
--out (default: mattbot_sim/results/fleet_<scenario>_<time>/robot<id>/). When every robot's run has ended,
the per-robot summaries are printed with their expectations; the exit code is 0 only if all passed.
Ctrl-C stops all robots (each still writes its summary).
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PKG, "src"))

from mattbot_sim.scenario import load_scenario  # noqa: E402

REAL_ROBOT_IDS_BELOW = 20  # the real fleet uses small ids; simulated robots should not reuse them


def scenario_path(name):
    if os.path.isfile(name):
        return os.path.abspath(name)
    path = os.path.join(PKG, "scenarios", name if name.endswith(".yaml") else name + ".yaml")
    if not os.path.isfile(path):
        sys.exit("no scenario %s" % name)
    return path


def latest_result(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.json")), key=os.path.getmtime)
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", help="name in mattbot_sim/scenarios or a path")
    ap.add_argument("--speed", type=float, default=1.0, help="simulated time speed (all robots)")
    ap.add_argument("--port-base", type=int, default=11450, help="robot k uses ROS master port port-base + k")
    ap.add_argument("--out", default="", help="output directory (logs and results per robot)")
    ap.add_argument("--timeout", type=float, default=0.0, help="wall seconds before stopping all robots (0: none)")
    args = ap.parse_args(argv)

    path = scenario_path(args.scenario)
    scenario = load_scenario(path)
    if not scenario.is_fleet:
        sys.exit("%s has no robots list; run single-robot scenarios with roslaunch mattbot_sim sim.launch" % path)
    low = [i for i in scenario.robot_ids if i < REAL_ROBOT_IDS_BELOW]
    if low:
        print("warning: robot ids %s are in the range of real robots; the sim's DDS is loopback-only, but prefer "
              "ids >= %d" % (low, REAL_ROBOT_IDS_BELOW))
    out = args.out or os.path.join(PKG, "results", "fleet_%s_%s" % (scenario.name, time.strftime("%Y%m%d_%H%M%S")))
    epoch = time.time()  # wall clock: shared simulated-time origin for every robot

    procs = []
    for k, rid in enumerate(scenario.robot_ids):
        rdir = os.path.join(out, "robot%d" % rid)
        os.makedirs(rdir, exist_ok=True)
        port = args.port_base + k
        env = dict(os.environ, ROS_MASTER_URI="http://localhost:%d" % port, ROS_HOSTNAME="localhost",
                   ROS_IP="127.0.0.1", ROBOT_ID=str(rid))
        env.pop("ROS_NAMESPACE", None)
        cmd = ["roslaunch", "-p", str(port), "mattbot_sim", "sim.launch", "scenario_file:=%s" % path,
               "robot_id:=%d" % rid, "dds:=true", "clock_epoch:=%.3f" % epoch, "speed:=%g" % args.speed,
               "results_dir:=%s" % rdir, "roadmap_cache_dir:=%s" % os.path.join(rdir, "roadmap")]
        cmd += ["%s:=%s" % kv for kv in scenario.launch_args.items()] + extra  # command line wins (later)
        log = open(os.path.join(rdir, "roslaunch.log"), "w")
        print("robot %d: ROS master :%d, log %s" % (rid, port, log.name), flush=True)
        procs.append((rid, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True), log))

    def stop(*_args):
        for _rid, p, _log in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)  # roslaunch shuts its nodes down; sim_monitor writes its summary

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("running %s with %d robots at %gx (fleet start %.0f s after launch); Ctrl-C stops all" % (
        scenario.name, len(procs), args.speed, scenario.fleet_start_s), flush=True)
    t0 = time.time()  # wall clock: --timeout
    stopping = False
    while any(p.poll() is None for _rid, p, _log in procs):
        if args.timeout > 0 and not stopping and time.time() - t0 > args.timeout:  # wall clock: --timeout
            print("timeout: stopping all robots", flush=True)
            stop()
            stopping = True
        # The first robot to finish its scenario ends the fleet: the others must not run on alone
        if not stopping and any(p.poll() is not None for _rid, p, _log in procs):
            stop()
            stopping = True
        time.sleep(1.0)  # wall clock: polling the child processes
    for _rid, _p, log in procs:
        log.close()

    all_ok = True
    print("\n===== fleet %s: %d robots, results in %s =====" % (scenario.name, len(procs), out))
    for rid, _p, _log in procs:
        res = latest_result(os.path.join(out, "robot%d" % rid))
        if res is None:
            print("robot %d: NO RESULTS (see its roslaunch.log)" % rid)
            all_ok = False
            continue
        s = res["summary"]
        checks = s.get("expectations", [])
        ok = s.get("expectations_passed", True)
        all_ok &= bool(ok)
        print("robot %d: %s  (run %.0f s sim / %.0f s wall; ledger added %d, removals %d correct / %d false, "
              "stale %d)" % (rid, "PASSED" if ok else "FAILED", s.get("run_s", 0), s.get("run_wall_s", 0),
                             s["ledger_objects_added"], s["correct_removals"], s["false_removals"],
                             len(s["stale_ledger_objects"])))
        for c in checks:
            print("    %s  %s %s %g (got %s)" % ("PASS" if c["ok"] else "FAIL", c["key"], c["op"], c["expected"],
                                               c["actual"]))
    print("===== fleet %s =====" % ("PASSED" if all_ok else "FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
