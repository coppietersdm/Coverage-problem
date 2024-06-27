#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/humble/setup.bash"
source "/coverage_crazyflie_ws/install/setup.bash"

exec "$@"