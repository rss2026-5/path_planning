#!/usr/bin/env bash
#
# deploy.sh — Sync path_planning to the RACECAR and optionally build inside Docker.
#
# Usage:
#   ./scripts/deploy.sh              # deploy to car 102 (default)
#   ./scripts/deploy.sh 104          # deploy to a different car number
#   ./scripts/deploy.sh --build      # deploy then build inside Docker on car 102
#   ./scripts/deploy.sh 104 --build  # deploy and build on a specific car
#
# Prerequisites:
#   - Connected to the car's WiFi
#   - Car running: cd && ./run_rostorch.sh  (starts the Docker container)
#   - Password: racecar@mit

set -euo pipefail

CAR_NUM="102"
DO_BUILD=false

for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=true ;;
        [0-9]*) CAR_NUM="$arg" ;;
    esac
done

CAR_HOST="racecar@192.168.1.${CAR_NUM}"
REMOTE_DIR="~/racecar_ws/src/path_planning"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Deploying path_planning to ${CAR_HOST}"
echo "    Source : ${REPO_ROOT}"
echo "    Target : ${CAR_HOST}:${REMOTE_DIR}"
echo ""

# Ensure remote directory exists
ssh "${CAR_HOST}" "mkdir -p ${REMOTE_DIR}"

# Rsync — never deletes bag_files or .db3 so recorded data is safe
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'bag_files' \
    --exclude '*.db3' \
    --exclude '*.pyc' \
    --exclude 'build' \
    --exclude 'install' \
    --exclude 'log' \
    "${REPO_ROOT}/" \
    "${CAR_HOST}:${REMOTE_DIR}/"

echo ""
echo "==> Files synced."

if [ "$DO_BUILD" = true ]; then
    echo ""
    echo "==> Building inside Docker on car (container: racecar)..."
    # colcon/ros2 only exist inside Docker — build via docker exec, not bare SSH
    ssh "${CAR_HOST}" \
        "docker exec racecar bash -lc 'cd /root/racecar_ws && colcon build --packages-select path_planning --symlink-install && echo BUILD_OK'"
    echo ""
    echo "==> Build complete. Source the workspace before launching:"
    echo "    source ~/racecar_ws/install/setup.bash"
else
    echo ""
    echo "==> Next steps (inside Docker on the car):"
    echo ""
    echo "    # Enter Docker (in a terminal that has SSH'd to the car):"
    echo "    connect"
    echo ""
    echo "    # Build (required after first deploy or any code change):"
    echo "    cd ~/racecar_ws && colcon build --packages-select path_planning --symlink-install && source install/setup.bash"
    echo ""
    echo "    # Launch everything (localization + planner + follower + safety):"
    echo "    ros2 launch path_planning real.launch.xml"
    echo ""
    echo "    # RViz (in a second Docker terminal, after export DISPLAY=:10):"
    echo "    export DISPLAY=:10"
    echo "    source ~/racecar_ws/install/setup.bash"
    echo "    rviz2"
    echo ""
    echo "    Or re-run with --build to build automatically:"
    echo "    ./scripts/deploy.sh ${CAR_NUM} --build"
fi

echo ""
echo "==> Done."
