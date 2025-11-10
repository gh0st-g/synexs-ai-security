#!/bin/bash
#
# Synexs Training Progress Monitor
# Monitors running training session by reading progress.json
#
# Usage: ./progress.sh [output_dir]
#

OUTPUT_DIR="${1:-./training_logs}"
PROGRESS_FILE="$OUTPUT_DIR/progress.json"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to format seconds to human readable
format_time() {
    local seconds=$1
    if (( $(echo "$seconds < 60" | bc -l) )); then
        printf "%.0fs" "$seconds"
    elif (( $(echo "$seconds < 3600" | bc -l) )); then
        printf "%.1fm" "$(echo "$seconds / 60" | bc -l)"
    else
        printf "%.1fh" "$(echo "$seconds / 3600" | bc -l)"
    fi
}

# Function to display progress
show_progress() {
    if [ ! -f "$PROGRESS_FILE" ]; then
        echo -e "${RED}Error: Progress file not found: $PROGRESS_FILE${NC}"
        echo "Make sure training is running and output directory is correct."
        exit 1
    fi

    # Read JSON data
    local data=$(cat "$PROGRESS_FILE")

    # Extract fields using python
    local mission_current=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('mission_current', 0))")
    local mission_total=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('mission_total', 0))")
    local progress=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('progress_percent', 0))")
    local elapsed=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('elapsed_seconds', 0))")
    local rate=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('rate_missions_per_sec', 0))")
    local eta=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('eta_seconds', 0))")
    local status=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    local timestamp=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('timestamp', ''))")

    # Stats
    local success=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('stats', {}).get('success_count', 0))")
    local failure=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('stats', {}).get('failure_count', 0))")
    local abort=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('stats', {}).get('abort_count', 0))")

    # Format times
    local elapsed_str=$(format_time $elapsed)
    local eta_str=$(format_time $eta)

    # Clear screen and show progress
    clear
    echo "========================================================================"
    echo -e "${BLUE}SYNEXS PHASE 1 - TRAINING PROGRESS${NC}"
    echo "========================================================================"
    echo ""

    # Status
    if [ "$status" = "running" ]; then
        echo -e "Status: ${GREEN}RUNNING${NC}"
    elif [ "$status" = "completed" ]; then
        echo -e "Status: ${GREEN}COMPLETED${NC}"
    elif [ "$status" = "interrupted" ]; then
        echo -e "Status: ${YELLOW}INTERRUPTED${NC}"
    else
        echo -e "Status: ${RED}UNKNOWN${NC}"
    fi

    echo -e "Last update: $timestamp"
    echo ""

    # Progress bar
    local bar_length=50
    local filled=$(echo "$progress / 100 * $bar_length" | bc -l | xargs printf "%.0f")
    local empty=$((bar_length - filled))

    echo -n "Progress: ["
    printf "${GREEN}%${filled}s${NC}" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    echo -e "] ${YELLOW}$(printf "%.1f" $progress)%%${NC}"
    echo ""

    # Mission progress
    echo -e "${BLUE}Missions${NC}"
    echo "  Current:  $mission_current / $mission_total"
    echo "  Remaining: $((mission_total - mission_current))"
    echo ""

    # Statistics
    echo -e "${BLUE}Results${NC}"
    echo "  Success:  $success"
    echo "  Failure:  $failure"
    echo "  Aborted:  $abort"
    if [ $mission_current -gt 0 ]; then
        local success_rate=$(echo "scale=1; $success * 100 / $mission_current" | bc -l)
        echo "  Success Rate: ${success_rate}%"
    fi
    echo ""

    # Timing
    echo -e "${BLUE}Timing${NC}"
    echo "  Elapsed:  $elapsed_str"
    echo "  Rate:     $(printf "%.2f" $rate) missions/sec"
    echo "  ETA:      $eta_str"
    echo ""

    echo "========================================================================"
    echo ""
    echo "Press Ctrl+C to stop monitoring (training will continue)"
    echo "Refresh: 2 seconds"
}

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}Error: Output directory not found: $OUTPUT_DIR${NC}"
    exit 1
fi

# Main monitoring loop
echo "Monitoring training progress..."
echo "Output directory: $OUTPUT_DIR"
echo ""

while true; do
    show_progress
    sleep 2
done
