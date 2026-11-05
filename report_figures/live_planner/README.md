# Live planner feedback

Choose a pick point in a recorded crate scan and see how each MoveIt planner would move from the capture pose to the
pre-grasp pose. Plans only, in a simulator on ROS master 11316. Nothing can be sent to the robot, and the server refuses
to start on the robot's master (11311).

    ./run_live.sh            # 10 cm fingers; ./run_live.sh 0.05 for 5 cm; Ctrl-C stops the simulator and the server
    LIVE_NICE=19 ./run_live.sh   # lower priority while the robot is in use

Then open http://localhost:8765. In VS Code, the Ports panel forwards this port if the browser runs outside the container.
The page loads three.js from a CDN, so the browser needs internet access. Don't run it during a benchmark: it changes the
benchmark's planning times.

The server uses the benchmark's own functions, so results compare with the 10 cm and 5 cm experiments: orientation choice
(test_crate_reachability, as simple_pick_point), crate contents carved as in the benchmark (`gripper`), nearest-IK joint
goal with the 180-degree roll, `plan_query` and `path_metrics`. Scans come from `experiments/hard_points/points_low.json`
(low edge and packed layouts) and the October points file.

Limits: only the capture to pre-grasp move, from saved scans, not the live camera. RRT* keeps its fixed 1 s and 5 s budgets
(the planning time setting applies to the other planners). A plan that has started can't be stopped, so a new request waits
for it. Changing finger length needs a restart.
