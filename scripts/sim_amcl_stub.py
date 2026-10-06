#!/usr/bin/env python3
"""Stand-in for amcl's dynamic_reconfigure server (the sim has perfect localization, no amcl).

The navigator reconfigures amcl after localizing (dynamic_reconfigure Client('amcl')); this node
accepts and ignores the update so that call succeeds. Launch it with name="amcl".
"""

import rospy
from amcl.cfg import AMCLConfig
from dynamic_reconfigure.server import Server


def on_reconfigure(config, _level):
    rospy.loginfo("sim_amcl_stub: ignoring amcl reconfigure (perfect localization in sim)")
    return config


if __name__ == "__main__":
    rospy.init_node("amcl")
    Server(AMCLConfig, on_reconfigure)
    rospy.spin()
