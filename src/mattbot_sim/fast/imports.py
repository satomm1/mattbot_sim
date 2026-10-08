"""Make the robot libraries importable without ROS (import this before anything from the stack).

- Puts mattbot_navigation/src, path_planning/src, mattbot_dds/src and mattbot_image_detection/src on sys.path
  (found next to mattbot_sim in the workspace, or via MATTBOT_SRC).
- dds_utils' package __init__ imports rospy and cyclonedds; only its pure submodules (ledger, belief) are
  used here, so a stub package is installed that skips the __init__ (as mattbot_navigation's tests do).
- Headless matplotlib (navigation_utils pulls it in through social_path_planning).
"""

import os
import sys
import types

SRC = os.environ.get("MATTBOT_SRC") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PATHS = ["mattbot_navigation/src", "path_planning/src", "mattbot_dds/src", "mattbot_image_detection/src"]

os.environ.setdefault("MPLBACKEND", "Agg")
for rel in PATHS:
    path = os.path.join(SRC, rel)
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

if "dds_utils" not in sys.modules or not hasattr(sys.modules["dds_utils"], "__path__"):
    _stub = types.ModuleType("dds_utils")
    _stub.__path__ = [os.path.join(SRC, "mattbot_dds", "src", "dds_utils")]
    sys.modules["dds_utils"] = _stub

MAP_JSON_DIR = os.path.join(SRC, "mattbot_mcl", "map_json")
