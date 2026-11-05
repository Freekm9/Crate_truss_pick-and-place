#!/bin/bash
# Starts an isolated sim (ROS master 11313, never the robot's 11311) with the E4 robot's 10 cm fingers, runs replan.py
# with the given arguments (it resumes), stops the sim by PID. Plans only: nothing is executed.
HERE=$(cd "$(dirname "$0")" && pwd)
source /opt/ros/noetic/setup.bash; source /root/Crate_truss_pick-and-place/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11313 GAZEBO_MASTER_URI=http://localhost:11347
mkdir -p /root/Crate_truss_pick-and-place/experiments/real_vs_sim_replans; SIMLOG=/root/Crate_truss_pick-and-place/experiments/real_vs_sim_replans/sim.log
roslaunch -p 11313 panda_moveit_config_vine_tomato gazebo_moveit.launch robot_name:=panda arm_id:=panda finger_extension_m:=0.10 tcp_xyz:="0 0 0.2034" scene_file:=samla_crate.scene rviz:=false headless:=true > "$SIMLOG" 2>&1 &
SIM=$!
stop_sim() { kill -INT $SIM 2>/dev/null; wait $SIM 2>/dev/null; }
trap 'stop_sim; exit 130' INT TERM
for i in $(seq 1 100); do grep -q "You can start planning now" "$SIMLOG" && break; sleep 3; done
grep -q "You can start planning now" "$SIMLOG" || { echo "sim did not start, see $SIMLOG"; stop_sim; exit 1; }
sleep 3
python3 "$HERE/replan.py" "$@"
RC=$?
stop_sim; echo "exit $RC"; exit $RC
