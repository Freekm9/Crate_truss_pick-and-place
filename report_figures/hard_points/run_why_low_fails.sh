#!/bin/bash
# Its own simulator on ROS master 11317 at the lowest priority (never the robot's 11311), stopped by PID afterwards.
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); source /opt/ros/noetic/setup.bash; source "$ROOT/devel/setup.bash"
export ROS_MASTER_URI=http://localhost:11317 GAZEBO_MASTER_URI=http://localhost:11351
LOG=$ROOT/experiments/hard_points/why_sim.log; rm -f "$ROOT/experiments/hard_points/why_low_fails.jsonl"
nice -n 19 roslaunch -p 11317 panda_moveit_config_vine_tomato gazebo_moveit.launch robot_name:=panda arm_id:=panda finger_extension_m:=0.10 tcp_xyz:="0 0 0.2034" scene_file:=samla_crate.scene rviz:=false headless:=true > "$LOG" 2>&1 &
SIM=$!; stop() { kill -INT $SIM 2>/dev/null; kill -TERM $SIM 2>/dev/null; wait $SIM 2>/dev/null; }; trap 'stop; exit 130' INT TERM
for i in $(seq 1 100); do grep -q "You can start planning now" "$LOG" && break; sleep 3; done; grep -q "You can start planning now" "$LOG" || { echo "sim did not start"; stop; exit 1; }
sleep 3; nice -n 19 python3 "$HERE/why_low_fails.py" "$@"; RC=$?; stop; echo "exit $RC"; exit $RC
