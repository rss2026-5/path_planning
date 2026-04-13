#!/usr/bin/env bash
#
# record_test.sh — Record all topics needed for the Lab 6 report.
#
# Run this ON THE CAR (inside Docker) AFTER launching:
#   ros2 launch path_planning real.launch.xml
#
# Usage:
#   ./scripts/record_test.sh                  # auto-timestamped name
#   ./scripts/record_test.sh my_run_name      # custom name
#
# Bags are saved to bag_files/<name>/ inside the package, which lives on the
# Docker volume mount and persists across container restarts.
#
# Copy bags to laptop:
#   scp -r racecar@192.168.1.102:~/racecar_ws/src/path_planning/bag_files/ ./bag_files/
#
# Topics recorded (covers all report requirements):
#   POSE / LOCALIZATION
#     /pf/pose/odom         — estimated pose in map frame (cross-track error, path tracking)
#     /pf/particles         — particle cloud (video: convergence visualization)
#     /vesc/odom            — raw wheel odometry (ground truth comparison)
#     /tf  /tf_static       — full transform tree
#
#   ENVIRONMENT
#     /map                  — occupancy grid (video: map overlay)
#     /scan                 — lidar scan (video: laser aligned with walls)
#
#   PLANNING & CONTROL
#     /trajectory/current   — planned trajectory as PoseArray
#     /goal_pose            — operator's nav goal clicks
#     /initialpose          — operator's pose estimate clicks
#     /vesc/input/navigation  — drive commands from pure pursuit
#     /vesc/low_level/input/safety  — safety controller interventions
#
#   VISUALIZATION (for video replay in RViz)
#     /planned_trajectory/path      — planned path marker
#     /planned_trajectory/start_point
#     /planned_trajectory/end_pose
#     /pure_pursuit/lookahead       — lookahead sphere marker

set -euo pipefail

NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BAG_DIR="${SCRIPT_DIR}/../bag_files/${NAME}"

mkdir -p "$(dirname "${BAG_DIR}")"

echo "==> Recording path_planning run to ${BAG_DIR}"
echo "    Press Ctrl-C to stop."
echo ""

ros2 bag record \
    /pf/pose/odom \
    /pf/particles \
    /vesc/odom \
    /tf \
    /tf_static \
    /map \
    /scan \
    /trajectory/current \
    /goal_pose \
    /initialpose \
    /vesc/input/navigation \
    /vesc/low_level/input/safety \
    /planned_trajectory/path \
    /planned_trajectory/start_point \
    /planned_trajectory/end_pose \
    /pure_pursuit/lookahead \
    -o "${BAG_DIR}"
