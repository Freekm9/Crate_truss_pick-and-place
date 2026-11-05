#!/bin/bash
# Live planner feedback: starts its own simulator (ROS master 11316, never the robot's 11311) and the page's server.
# Plans only; Ctrl-C stops both.
#   run_live.sh [FINGER_M]         default 0.10, the robot's fingers; 0.05 for the 5 cm setup
#   LIVE_NICE=19 run_live.sh       lower priority while the robot is in use (default 0)
FINGER_M=${1:-0.10}; TCP_Z=$(python3 -c "print(f'{0.1034 + $FINGER_M:.4f}')"); NICE=${LIVE_NICE:-0}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
source /opt/ros/noetic/setup.bash; source "$ROOT/devel/setup.bash"
export ROS_MASTER_URI=http://localhost:11316 GAZEBO_MASTER_URI=http://localhost:11350
if timeout 3 rosnode list >/dev/null 2>&1; then echo "ROS master 11316 is already in use: is the live planner already running?"; exit 1; fi
SIMLOG=$ROOT/experiments/live_planner_sim.log
nice -n $NICE roslaunch -p 11316 panda_moveit_config_vine_tomato gazebo_moveit.launch robot_name:=panda arm_id:=panda finger_extension_m:=$FINGER_M tcp_xyz:="0 0 $TCP_Z" scene_file:=samla_crate.scene rviz:=false headless:=true > "$SIMLOG" 2>&1 &
SIM=$!
SERVER=""
stop() { [ -n "$SERVER" ] && kill -TERM $SERVER 2>/dev/null; kill -INT $SIM 2>/dev/null; kill -TERM $SIM 2>/dev/null; wait $SIM 2>/dev/null; exit ${1:-0}; }  # a background job ignores SIGINT, so TERM too
trap 'stop 130' INT TERM
echo "Starting the simulator with $FINGER_M m fingers (about 30 s)..."
for i in $(seq 1 100); do grep -q "You can start planning now" "$SIMLOG" && break; sleep 3; done
grep -q "You can start planning now" "$SIMLOG" || { echo "The simulator did not start, see $SIMLOG"; stop 1; }
sleep 3
nice -n $NICE python3 "$HERE/server.py" &
SERVER=$!
wait $SERVER; stop $?
