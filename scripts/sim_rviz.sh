#!/usr/bin/env bash
# Open RViz with the sim's display config, fetched from the running sim's ROS master (/sim/rviz_config).
# Needs only ROS Noetic + RViz on this machine, not the mattbot_sim package. Set ROS_MASTER_URI / ROS_IP first.
#   ./sim_rviz.sh [extra rviz args]
set -e
CONFIG="${HOME}/.rviz/mattbot_sim.rviz"
mkdir -p "$(dirname "$CONFIG")"
if ! python3 -c 'import sys, rospy; sys.stdout.write(rospy.get_param("/sim/rviz_config"))' > "$CONFIG.tmp" 2>/dev/null; then
    rm -f "$CONFIG.tmp"
    echo "Could not read /sim/rviz_config from ${ROS_MASTER_URI:-<ROS_MASTER_URI unset>}; is sim.launch running?" >&2
    if [ -f "$CONFIG" ]; then
        echo "Using the last fetched config: $CONFIG" >&2
    else
        exit 1
    fi
else
    mv "$CONFIG.tmp" "$CONFIG"
fi
exec rviz -d "$CONFIG" "$@"
