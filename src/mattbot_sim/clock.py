"""ROS time helpers for the sim nodes (the only module here that touches rospy; imported lazily).

With /use_sim_time, ROS time is 0 until sim_world publishes the first /clock; nodes that read the time
while starting must wait for it. Without /use_sim_time, ROS time is the wall clock and this returns at once.
"""

import time


def wait_for_clock(poll_s=0.05):
    """Block until ROS time is valid (> 0). Call after rospy.init_node. Returns the current ROS time."""
    import rospy

    while not rospy.is_shutdown() and rospy.get_time() <= 0.0:
        time.sleep(poll_s)  # wall clock: ROS time is not running yet
    return rospy.get_time()
