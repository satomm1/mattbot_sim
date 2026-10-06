cmake_minimum_required(VERSION 3.0.2)
project(mattbot_sim)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/sim_world.py
  scripts/sim_amcl_stub.py
  scripts/scenario_runner.py
  scripts/sim_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
