#!/bin/bash
# Synexs Defensive Security Training - Quick Start Script
# Launches full defensive training system

set -e

echo "============================================================"
echo "🛡️  Synexs Defensive Security Training System"
echo "============================================================"
echo "⚠️  All traffic is LOCAL ONLY (127.0.0.1)"
echo "🎯 Purpose: Defensive security education"
echo "============================================================"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import flask" 2>/dev/null || {
    echo "❌ Flask not found. Installing..."
    pip install flask
}

python3 -c "import requests" 2>/dev/null || {
    echo "❌ Requests not found. Installing..."
    pip install requests
}

echo "✅ Dependencies OK"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p datasets/honeypot datasets/agents
echo "✅ Directories ready"
echo ""

# Start honeypot
echo "🍯 Starting honeypot server (127.0.0.1:8080)..."
python3 honeypot_server.py > honeypot.log 2>&1 &
HONEYPOT_PID=$!
sleep 2

# Check honeypot
if ! curl -s http://127.0.0.1:8080 > /dev/null; then
    echo "❌ Honeypot failed to start"
    kill $HONEYPOT_PID 2>/dev/null
    exit 1
fi
echo "✅ Honeypot running (PID: $HONEYPOT_PID)"
echo ""

# Start listener
echo "🎧 Starting listener (127.0.0.1:8443)..."
python3 listener.py > listener.log 2>&1 &
LISTENER_PID=$!
sleep 2
echo "✅ Listener running (PID: $LISTENER_PID)"
echo ""

# Spawn agents
echo "🚀 Spawning defensive training agents..."
python3 propagate_v3.py
echo ""

# Run agents
echo "⚔️  Launching agents (attacking honeypot)..."
for agent in datasets/agents/sx*.py; do
    [ -f "$agent" ] || continue
    python3 "$agent" &
done

sleep 5
echo ""

# Show stats
echo "============================================================"
echo "📊 Initial Statistics"
echo "============================================================"
curl -s http://127.0.0.1:8080/stats | python3 -m json.tool 2>/dev/null || echo "Stats not yet available"
echo ""

echo "============================================================"
echo "✅ Defensive Training System RUNNING"
echo "============================================================"
echo "📍 Honeypot: http://127.0.0.1:8080"
echo "📊 Stats:    http://127.0.0.1:8080/stats"
echo "📝 Logs:     honeypot.log, listener.log"
echo "🔍 Monitor:  tail -f listener.log"
echo ""
echo "To stop:"
echo "  kill $HONEYPOT_PID $LISTENER_PID"
echo "  pkill -f 'python3 datasets/agents'"
echo ""
echo "📚 Read DEFENSIVE_TRAINING.md for full documentation"
echo "============================================================"
