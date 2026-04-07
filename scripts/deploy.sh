#!/usr/bin/env bash
#
# deploy.sh — Sync path_planning package to the RACECAR
#
# Usage:
#   ./scripts/deploy.sh              # deploy to default car (192.168.1.102)
#   ./scripts/deploy.sh 192.168.1.X  # deploy to a specific car IP
#   ./scripts/deploy.sh --build      # deploy and build on the car
#
# Prerequisites:
#   - Connected to the car's WiFi network
#   - SSH key or password for racecar@<car-ip> (default password: racecar@mit)

set -euo pipefail

CAR_IP="${1:-192.168.1.102}"
DO_BUILD=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=true ;;
        [0-9]*) CAR_IP="$arg" ;;
    esac
done

CAR_USER="racecar"
CAR_HOST="${CAR_USER}@${CAR_IP}"
REMOTE_PKG_DIR="~/racecar_ws/src/path_planning"

# Resolve the repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Deploying path_planning ==="
echo "  Source:  ${REPO_ROOT}"
echo "  Target:  ${CAR_HOST}:${REMOTE_PKG_DIR}"
echo ""

# Test SSH connectivity
echo "Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${CAR_HOST}" true 2>/dev/null; then
    echo "Cannot reach ${CAR_HOST} with key auth."
    echo "Trying with password prompt (password: racecar@mit)..."
    ssh -o ConnectTimeout=5 "${CAR_HOST}" true
fi

# Rsync the package
echo "Syncing files..."
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
    "${CAR_HOST}:${REMOTE_PKG_DIR}/"

echo ""
echo "Files synced successfully."

# Optionally build on the car
if [ "$DO_BUILD" = true ]; then
    echo ""
    echo "=== Building on car ==="
    ssh "${CAR_HOST}" bash -lc "'
        export SIM_WS=/root/sim_ws
        cd ~/racecar_ws && \
        colcon build --packages-select path_planning --symlink-install && \
        source install/setup.bash && \
        echo \"Build succeeded.\"
    '"
else
    echo ""
    echo "To build on the car, run:"
    echo "  ssh ${CAR_HOST}"
    echo "  export SIM_WS=/root/sim_ws"
    echo "  cd ~/racecar_ws && colcon build --packages-select path_planning --symlink-install && source install/setup.bash"
    echo ""
    echo "Or re-run with: ./scripts/deploy.sh --build"
fi

echo ""
echo "=== Done ==="
