#!/bin/bash
###############################################################################
# Synexs Production System Startup Script
#
# This script launches the current production Synexs system:
# - Core orchestrator (cellular architecture)
# - Honeypot server (WAF + AI detection)
# - AI swarm (learning engine)
# - Listener (kill reports)
# - Continuous training (attack generation)
#
# Usage:
#   ./start_biological_organism.sh
#
# The system will:
# 1. Stop any existing Synexs processes
# 2. Start all production services
# 3. Begin continuous training data collection
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  SYNEXS PRODUCTION SYSTEM - STARTUP"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Working Directory: $SCRIPT_DIR"
echo "Python: $(which python3)"
echo "Virtual Env: $SCRIPT_DIR/synexs_env/bin/python3"
echo ""

# Use virtual environment python if available
PYTHON_BIN="$SCRIPT_DIR/synexs_env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    echo "⚠️  Virtual environment not found, using system python3"
    PYTHON_BIN="python3"
fi

# Check if production system files exist
echo "📋 Checking production files..."
REQUIRED_FILES=(
    "honeypot_server.py"
    "listener.py"
    "ai_swarm_fixed.py"
    "synexs_core_orchestrator.py"
    "propagate_v4.5.py"
    "attack_profiles.json"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "❌ ERROR: $MISSING required files missing"
    exit 1
fi

echo "✓ All production files present"
echo ""

# Stop existing Synexs processes
echo "🛑 Stopping existing Synexs processes..."

# Stop honeypot
if pgrep -f "honeypot_server.py" > /dev/null; then
    echo "  Stopping honeypot_server..."
    pkill -f "honeypot_server.py" || true
fi

# Stop swarm
if pgrep -f "ai_swarm_fixed.py" > /dev/null; then
    echo "  Stopping ai_swarm_fixed..."
    pkill -f "ai_swarm_fixed.py" || true
fi

# Stop listener
if pgrep -f "listener.py" > /dev/null; then
    echo "  Stopping listener..."
    pkill -f "listener.py" || true
fi

# Stop orchestrator
if pgrep -f "synexs_core_orchestrator.py" > /dev/null; then
    echo "  Stopping orchestrator..."
    pkill -f "synexs_core_orchestrator.py" || true
fi

# Stop continuous training
if pgrep -f "start_continuous_training.sh" > /dev/null; then
    echo "  Stopping continuous training..."
    pkill -f "start_continuous_training.sh" || true
fi

sleep 2
echo "✓ Existing processes stopped"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p datasets/honeypot
mkdir -p datasets/generated
mkdir -p datasets/refined
mkdir -p datasets/decisions
mkdir -p datasets/agents
mkdir -p datasets/ai_decisions
echo "✓ Directories ready"
echo ""

# Start production services
echo "════════════════════════════════════════════════════════════════════"
echo "  STARTING PRODUCTION SERVICES"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# 1. Start honeypot
echo "🍯 Starting honeypot server (port 8080)..."
nohup $PYTHON_BIN honeypot_server.py > /dev/null 2>&1 &
sleep 2

if pgrep -f "honeypot_server.py" > /dev/null; then
    echo "  ✅ Honeypot running (PID: $(pgrep -f "honeypot_server.py"))"
else
    echo "  ❌ Honeypot failed to start"
    exit 1
fi

# 2. Start listener
echo "📡 Starting listener (port 5555)..."
nohup $PYTHON_BIN listener.py > /dev/null 2>&1 &
sleep 1

if pgrep -f "listener.py" > /dev/null; then
    echo "  ✅ Listener running (PID: $(pgrep -f "listener.py"))"
else
    echo "  ⚠️  Listener may have failed"
fi

# 3. Start AI swarm
echo "🤖 Starting AI swarm..."
nohup $PYTHON_BIN ai_swarm_fixed.py > /dev/null 2>&1 &
sleep 1

if pgrep -f "ai_swarm_fixed.py" > /dev/null; then
    echo "  ✅ AI Swarm running (PID: $(pgrep -f "ai_swarm_fixed.py"))"
else
    echo "  ⚠️  AI Swarm may have failed"
fi

# 4. Start orchestrator
echo "🎯 Starting core orchestrator..."
nohup $PYTHON_BIN synexs_core_orchestrator.py > /dev/null 2>&1 &
sleep 2

if pgrep -f "synexs_core_orchestrator.py" > /dev/null; then
    echo "  ✅ Orchestrator running (PID: $(pgrep -f "synexs_core_orchestrator.py"))"
else
    echo "  ❌ Orchestrator failed to start"
    exit 1
fi

echo ""
echo "✅ All production services started"
echo ""

# 5. Start continuous training (optional)
echo "════════════════════════════════════════════════════════════════════"
echo "  CONTINUOUS TRAINING DATA COLLECTION"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Would you like to start continuous training data collection?"
echo "This will generate diverse attack patterns every 5 minutes using propagate_v4.5.py"
echo ""
read -p "Start continuous training? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Starting continuous training with propagate_v4.5..."

    # Check if start_continuous_training.sh exists, otherwise create inline loop
    if [ -f "start_continuous_training.sh" ]; then
        nohup ./start_continuous_training.sh > continuous_training.log 2>&1 &
    else
        echo "⚠️  start_continuous_training.sh not found, creating inline training loop..."
        nohup bash -c "while true; do $PYTHON_BIN propagate_v4.5.py >> continuous_training.log 2>&1; sleep 300; done" &
    fi

    sleep 2

    if pgrep -f "propagate_v4.5.py\|start_continuous_training.sh" > /dev/null; then
        echo "  ✅ Continuous training running"
        echo "  📋 Logs: tail -f continuous_training.log"
        echo "  📊 Attack logs: tail -f datasets/logs/attacks_log.jsonl"
    else
        echo "  ❌ Continuous training failed to start"
    fi
else
    echo ""
    echo "⏭️  Skipping continuous training"
    echo "   To start manually: $PYTHON_BIN propagate_v4.5.py"
    echo "   Or for continuous loop: while true; do $PYTHON_BIN propagate_v4.5.py; sleep 300; done &"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  SYNEXS PRODUCTION SYSTEM - RUNNING"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 System Status:"
echo "  ✅ Honeypot: http://127.0.0.1:8080"
echo "  ✅ Core Services: Running"
echo "  ✅ Data Collection: Active"
echo ""
echo "📋 Monitoring Commands:"
echo "  • System status:     ps aux | grep -E 'honeypot|swarm|orchestrator|listener'"
echo "  • Orchestrator:      tail -f synexs_core.log"
echo "  • Honeypot attacks:  tail -f datasets/honeypot/attacks.json"
echo "  • Training attacks:  tail -f datasets/logs/attacks_log.jsonl"
echo "  • AI decisions:      tail -f ai_decisions_log.jsonl"
echo "  • Attack stats:      cat datasets/logs/attacks_log.jsonl | jq -r '.attack_type' | sort | uniq -c"
echo "  • Generated agents:  ls -lh datasets/agents/ | wc -l"
echo ""
echo "🛑 To stop all services:"
echo "   pkill -f 'honeypot_server.py|listener.py|ai_swarm|orchestrator|propagate_v4.5|continuous_training'"
echo ""
echo "════════════════════════════════════════════════════════════════════"
