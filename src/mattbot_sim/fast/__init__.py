"""Fast, ROS-free simulator of the mattbot observation stack (see README "Fast simulator").

The robots' decisions come from the same pure libraries the ROS nodes call (navigation_utils,
dds_utils.ledger/belief, observation_eval.evidence, mattbot_sim world/perception/scoring). This package
only re-implements the ROS glue: the navigator state machine, mapper, evaluator windows, ledger and
planner nodes, with kinematic motion and one global clock. Import mattbot_sim.fast.imports first.
"""
