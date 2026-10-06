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
| `observe_dwell_s`, `observe_check_below_belief`, `observe_min_absent_frames` | `4`, `0.5`, `8` | as in bringup |
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

RViz is not installed in the robot container. Run the sim with a reachable IP, then point RViz at that master:

```bash
# on the Jetson
export ROS_MASTER_URI=http://<jetson-ip>:11411 ROS_IP=<jetson-ip>
roslaunch mattbot_sim sim.launch scan:=true
# on a laptop with ROS Noetic + RViz (and this package, or just copy rviz/sim.rviz)
export ROS_MASTER_URI=http://<jetson-ip>:11411 ROS_IP=<laptop-ip>
rviz -d sim.rviz
```

The config shows:
- the map and `/navigation_map`
- the true objects, green when present and grey when removed
- the objects the mapper has confirmed
- observation viewsheds and stops
- the smoothed path, the true pose and the scan

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
```

Objects can be placed anywhere free, including in the robot's path. This needs mattbot_navigation `98ab93a`
("Fix replan loop when a confirmed object is near the planned path") or later. Before that commit, the planner did not see
object blockouts, while `path_still_valid` did, so an object whose blockout overlapped the planned path made the navigator
replan the same path over and over ("Path no longer valid..."). The shipped scenarios keep objects against the walls,
so they also run on older navigator versions.

## Results

`sim_monitor` compares `/object_beliefs` (the active ledger objects) and `/observation/results` with the ground truth
(`/sim/ground_truth`). The summary includes:

- `correct_removals` / `false_removals`, and `removal_latency_s` (from the moment the object was taken away until the ledger dropped it)
- `stale_ledger_objects`: objects still in the ledger that are no longer there
- `spurious_ledger_objects`: ledger objects with no true object nearby
- `observation_stops` (STARTED / ENDED / ABORTED) and `observation_outcomes` (PRESENT / ABSENT / INCONCLUSIVE), plus `wrong_observation_outcomes`
- `blockouts_still_present` / `blockout_clear_after_s`: whether the mapper's `/object_map` blockout cleared after a ledger removal (today it only expires by TTL)

## Tests

```bash
python3 -m pytest mattbot_sim/test        # from src/, no ROS needed
```
