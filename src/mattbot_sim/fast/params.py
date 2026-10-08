"""Fast-sim parameters. Names match mattbot_sim/launch/sim.launch args (defaults = its values), so the
same names work in a scenario's launch_args, on the command line and in sweeps. Model constants copied
from the robot code are below, with their source."""

from dataclasses import dataclass, fields


@dataclass
class FastParams:
    # ---- sim.launch args ----
    cruising_velocity: float = 0.40
    belief_hold_s: float = 20.0
    belief_decay_s: float = 40.0
    observe: bool = True
    observe_dwell_s: float = 4.0
    observe_check_below_belief: float = 1.0
    observe_cooldown_s: float = 60.0
    observe_min_absent_frames: int = 8
    observe_detour: bool = False
    observe_max_detour_m: float = 15.0
    observe_detour_n_trips: float = 5.0
    observe_detour_margin_m: float = 0.0
    observe_detour_timeout_factor: float = 2.0
    miss_prob: float = 0.0
    pos_noise_std: float = 0.0
    false_pos_rate: float = 0.0
    detect_hz: float = 5.0
    ledger_blockout: bool = False  # mapper's enable_ledger_blockout (off in sim.launch)
    # ---- fast-sim model ----
    dt: float = 0.1  # s; navigator loop (10 Hz, localize_and_navigate.py run())
    block_wait_s: float = 3.0  # blocked by another robot this long -> replan around it
    robot_radius: float = 0.2  # sim_world collision disc (also how other robots see / block it)

    # ---- constants from the robot code ----
    v_max: float = 0.7  # localize_and_navigate.py v_max (sim_world clamps v the same)
    w_max: float = 2.5  # sim_world w clamp (heading controller asks up to 3)
    near_thresh: float = 0.35  # navigator near_goal
    theta_start_thresh: float = 0.05  # navigator aligned()
    theta_goal_thresh: float = 0.05
    park_pose_thresh: float = 0.05
    post_align_pause_s: float = 1.0
    robot_clearance: float = 0.4  # navigation launch robot_clearance -> robot_d = 0.8 for A*
    path_check_tolerance_m: float = 0.1
    observe_trigger_m: float = 0.25
    observe_path_stride: int = 5
    observe_turn_timeout_s: float = 10.0
    stop_passed_slack_s: float = 2.0  # navigator _observation_due: entry t + 2 s
    soft_start_s: float = 1.5  # TrajectoryTracker soft start (reference lags 0.75 s after it)
    mapper_confirm_count: int = 6  # occupancy_grid_mapper: confirmed when seen > 5 times
    mapper_miss_limit: int = 10
    mapper_ttl_s: float = 30.0  # confirmed-object blockouts expire (only while IDLE)
    mapper_max_blockout_m: float = 0.5
    ledger_match_radius_m: float = 0.75
    resight_min_interval_s: float = 30.0
    r_min: float = 1.0  # observation_planner viewsheds
    r_max: float = 3.5

    def update(self, values):
        """Set fields from a {name: value} mapping (strings are converted to the field's type)."""
        types_ = {f.name: f.type for f in fields(self)}
        for name, value in values.items():
            if name not in types_:
                continue  # e.g. scenario launch_args meant only for the ROS sim
            kind = types_[name]
            if kind in ("bool", bool):
                value = value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes")
            elif kind in ("int", int):
                value = int(float(value))
            elif kind in ("float", float):
                value = float(value)
            setattr(self, name, value)
        return self

    @staticmethod
    def names():
        return [f.name for f in fields(FastParams)]
