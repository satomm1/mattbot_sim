"""Make src/mattbot_sim importable without a catkin build: python3 -m pytest mattbot_sim/test"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
