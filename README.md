# mattbot_sim

A lightweight 2D simulator for testing the mattbot stack's **opportunistic obstacle checking** without hardware.
Only the hardware, localization and perception are simulated. The real nodes run unchanged:
navigator, observation planner, observation evaluator, ledger and belief.

| Real robot | In simulation |
|---|---|
| `mcu_comms.py` (motors/odometry) | `sim_world.py`: unicycle model driven by `/cmd_vel`; publishes `/odom` and TF `odom → base_footprint` |
| amcl, `localization_quality.py` | perfect localization: `sim_world.py` publishes TF `map → odom` so `map → base_footprint` is ground truth. `sim_amcl_stub.py` only answers the navigator's amcl dynamic_reconfigure call |
| Astra camera + OSOD detector | `sim_world.py`: perfect detections on `/detected_objects` of objects in the camera's FOV (≈60°), within 5 m and in line of sight; frames are dropped while turning, as the real detector does. Synthetic `/camera/color/camera_info` |
| `entry_exit.py` map loading | `sim_world.py` publishes latched `/map`, `/map_mod`, `/map_metadata` |
| rplidar | optional `/scan` (`scan:=true`), for RViz only |

The real nodes that run: `twist_mux`, `publish_transforms.py` (short URDF), `occupancy_grid_mapper`, `navigator_node` (observe on),
`observation_planner`, `observation_evaluator` (depth check off), `observation_ledger`, `object_belief_map`.
No DDS nodes run. The ledger works without them.

## Run

```bash
cd /workspace/catkin_ws && catkin_make && source devel/setup.bash

# Always use a separate ROS master: the sim publishes /cmd_vel. sim_world refuses to start if it sees the real drivers.
export ROS_MASTER_URI=http://localhost:11411
roslaunch mattbot_sim sim.launch scenario:=remove_one
```

The run proceeds as follows:
1. The robot "localizes" (about 20 s; these are the navigator's own fixed LOCALIZING delays).
2. `scenario_runner` patrols the scenario's waypoints and fires its timed events.
3. `sim_monitor` logs every scored event.
4. When the scenario's `duration` is reached (or on Ctrl-C), `sim_monitor` prints a summary and writes `results/<scenario>_<time>.json`.

Useful launch args:

| Arg | Default | |
|---|---|---|
| `scenario` | `remove_one` | file in `scenarios/` (or `scenario_file:=/abs/path.yaml`) |
| `duration` | `-1` | run length in s after localization; `-1` = scenario's value, `0` = until Ctrl-C |
| `belief_hold_s`, `belief_decay_s` | `20`, `40` | faster than the robot defaults (60, 120) so checks happen within minutes |
| `observe_cooldown_s` | `60` | per-object re-check cooldown in the planner (robot default 120) |
| `observe_dwell_s`, `observe_check_below_belief`, `observe_min_absent_frames` | `4`, `1.0`, `8` | dwell and absent frames as in bringup; the belief threshold is higher than bringup's 0.5, so every due object is checked |
| `observe_detour` | `false` | let the robot leave its path to check objects no path point can see (see [Detours](#detours)) |
| `observe_max_detour_m`, `observe_detour_n_trips`, `observe_detour_margin_m` | `15`, `5`, `0` | detour value rule, as in the navigation launch |
| `observe_detour_timeout_factor` | `2` | the navigator abandons a detour after factor × estimated cost + 20 s |
| `roadmap_cache_dir` | `~/.ros/mattbot_roadmap` | observation_planner's roadmap / importance cache |
| `miss_prob`, `pos_noise_std`, `false_pos_rate` | `0` | make the detector imperfect (per-object miss probability, metres, expected false detections per frame) |
| `scan` | `false` | publish a simulated lidar scan for RViz |

### Change the world while it runs

```bash
rostopic pub -1 /sim/event std_msgs/String "data: 'remove: chair_1'"
rostopic pub -1 /sim/event std_msgs/String "data: 'restore: chair_1'"
rostopic pub -1 /sim/event std_msgs/String "data: '{move: chair_1, x: 38.0, y: 19.7}'"
rostopic pub -1 /sim/event std_msgs/String "data: '{add: {id: cone_9, class: cone, x: 41.0, y: 18.15, width: 0.3}}'"
```

You can also send goals yourself with `/external_goal` (`geometry_msgs/Pose2D`) or with RViz "2D Nav Goal". Use a scenario without waypoints if you don't want the runner to send any.

### Watch in RViz (from another machine)

RViz is not installed in the robot container. Run the sim with a reachable IP. `sim.launch` puts the RViz config
([rviz/sim.rviz](rviz/sim.rviz)) on the parameter server as `/sim/rviz_config`, and [scripts/sim_rviz.sh](scripts/sim_rviz.sh)
fetches it and opens RViz with every display already set up:

```bash
# on the Jetson
export ROS_MASTER_URI=http://<jetson-ip>:11411 ROS_IP=<jetson-ip>
roslaunch mattbot_sim sim.launch scan:=true
# on a laptop with ROS Noetic + RViz. The package is not needed; copy sim_rviz.sh once, e.g.
#   scp <user>@<jetson-ip>:/workspace/catkin_ws/src/mattbot_sim/scripts/sim_rviz.sh ~/bin/
export ROS_MASTER_URI=http://<jetson-ip>:11411 ROS_IP=<laptop-ip>
sim_rviz.sh
```

Without the script, this one-liner does the same:
`python3 -c 'import sys, rospy; sys.stdout.write(rospy.get_param("/sim/rviz_config"))' > /tmp/sim.rviz && rviz -d /tmp/sim.rviz`.
The fetched config is saved to `~/.rviz/mattbot_sim.rviz` and replaced on every run. That file is also used when the master
can't be reached. To keep display changes, edit `rviz/sim.rviz` in this package.

The config shows:
- the map and `/navigation_map`
- the true objects, green when present and grey when removed
- the objects the mapper has confirmed
- observation viewsheds and stops
- the smoothed path, the true pose and the scan
- disabled (tick to show): the object belief map, the A* planned path, waypoints, object blockouts (`/object_map`), and the observation roadmap and detours

## Scenarios

`scenarios/*.yaml` (coordinates are in the robot map `current_map`, around its main corridor):

- `remove_one`: three objects; `chair_1` is removed at t = 150 s. Expected: a later check reports ABSENT and the ledger removes it, while the other objects are re-confirmed.
- `moved_object`: `chair_1` moves 3 m at t = 150 s. Expected: the old entry is removed and a new chair is added.
- `occluded`: `cone_1` sits behind `bench_1` as seen from the west, and is removed at t = 150 s. Expected: no false removals.

Format (see `src/mattbot_sim/scenario.py`):

```yaml
name: my_test
map: current_map          # mattbot_mcl/map_json stem (with its _mod variant), or a path to a map_server .yaml
start: {x: 28.0, y: 18.95, theta: 0.0}
objects:
  - {id: chair_1, class: chair, x: 35.0, y: 19.7, width: 0.4}   # use detector class names (not person/unknown)
waypoints:
  - {x: 46.0, y: 18.95, theta: 3.14159}
loop: true
duration: 900             # s after localization; 0 = until Ctrl-C
events:                   # s after localization
  - {at: 150, remove: chair_1}
# optional: peer_id, per-object known_from_peer / detour, and expect (see Detours and Results)
```

Objects can be placed anywhere free, including in the robot's path. This needs mattbot_navigation `98ab93a`
("Fix replan loop when a confirmed object is near the planned path") or later. Before that commit, the planner did not see
object blockouts, while `path_still_valid` did, so an object whose blockout overlapped the planned path made the navigator
replan the same path over and over ("Path no longer valid..."). The shipped scenarios keep objects against the walls,
so they also run on older navigator versions.

## Detours

With `observe_detour:=true`, the observation planner may also send the robot off its path. This only
happens for a ledger object that no point of the path can see, and only when the extra driving pays off:
V = N · (1 − belief)/2 · I_o must beat C = detour + v · (turn + dwell). I_o is the object's importance,
the extra travel per trip if the object blocks a passage. The navigator drives to the viewpoint, looks,
and replans to its goal. See `mattbot_navigation/scripts/observation_planner.py` and `KNOWN_ISSUES.md`.

Because such an object is hidden from the patrol, the robot can't add it to its ledger itself. Mark it
`known_from_peer: true`, and `scenario_runner` reports it at the start on `/ledger/observation_from_agent`
as if robot `peer_id` (default 98) had seen it, which is how the fleet would learn about it.

- `detour_present`: `cone_9` blocks the passage north of the corridor's west end (I_o ≈ 9 m). Expected: a
  detour of about 8 m to around (26.1, 21.5) once its belief has decayed, a PRESENT check, and the
  patrol resumes. `cone_1` is an opportunistic control.
- `detour_removed`: the same cone is taken away at t = 150 s. Expected: a later detour check reports
  ABSENT and the ledger removes it. Run it with `observe_detour:=false` for the baseline: the cone is never
  checked and stays in the ledger (`stale_ledger_objects`), so its expectations fail.
- `detour_not_worth_it`: a cone in the dead-end room south of the east end (I_o = 0). Expected: no detour.

```bash
roslaunch mattbot_sim sim.launch scenario:=detour_present observe_detour:=true
```

The first run builds the roadmap (about 1 s, then cached). To view the detours in RViz, tick
"Observation detours" and "Observation roadmap" in `sim.rviz`.

**Before a run, check a scenario offline.** No ROS master is needed. The checker scores each object on each
patrol leg the way the planner does, with belief 0:

```bash
rosrun mattbot_sim check_detour_scenario.py $(rospack find mattbot_sim)/scenarios/detour_present.yaml [--n-trips 5]
```

It exits 1 in two cases: an object annotated `detour: expected` (or `declined`) would not get (or would get)
a detour, or the sim camera could see the object from the path. In that second case the robot could find
it while patrolling. `test/test_detour_scenarios.py` runs the same check on every annotated scenario.

## Results

`sim_monitor` compares `/object_beliefs` (the active ledger objects) and `/observation/results` with the ground truth
(`/sim/ground_truth`). The summary includes:

- `correct_removals` / `false_removals`, and `removal_latency_s` (from the moment the object was taken away until the ledger dropped it)
- `stale_ledger_objects`: objects still in the ledger that are no longer there
- `spurious_ledger_objects`: ledger objects with no true object nearby
- `observation_stops` (STARTED / ENDED / ABORTED) and `observation_outcomes` (PRESENT / ABSENT / INCONCLUSIVE), plus `wrong_observation_outcomes`
- `blockouts_still_present` / `blockout_clear_after_s`: whether the mapper's `/object_map` blockout cleared after a ledger removal (today it only expires by TTL)
- `observation_stops_by_kind` / `observation_outcomes_by_kind` (OPPORTUNISTIC / DETOUR), and `detour_stops` (DETOUR stops ENDED). The `stops` list in the JSON has every stop event with the robot's position.
- `detours_started` / `detours_reached` / `detours_done` / `detours_abandoned` / `detours_cancelled`, plus `detour_abandon_reasons`. These are read from the navigator's log on `/rosout_agg`, because abandoned and cancelled detours publish no ObservationEvent.
- `goal_resends` / `goals_given_up`: how often `scenario_runner` had to re-send a goal because the navigator went idle short of it. A re-send can hide a navigator that never replanned, for example after a detour.

A scenario's optional `expect:` block is checked at the end of the run. The summary prints PASS/FAIL per key and stores it under `summary.expectations`. Keys are summary fields, with dotted paths for nested ones and lists counted by length. An optional `min_` / `max_` prefix turns the check into a lower or upper bound; without a prefix it checks equality:

```yaml
expect:
  min_detour_stops: 1
  min_observation_outcomes_by_kind.DETOUR.PRESENT: 1
  false_removals: 0
  goal_resends: 0
```

## Tests

```bash
python3 -m pytest mattbot_sim/test        # from src/, no ROS needed
```

`test_detour_scenarios.py` also needs `mattbot_navigation/src` and `path_planning/src`. It adds them itself
and is skipped if they can't be imported.
