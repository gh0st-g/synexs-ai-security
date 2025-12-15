#!/bin/bash
# Synexs Auto-Start Script
# Starts all critical processes on boot

SYNEXS_DIR="/root/synexs"
PYTHON="/root/synexs/synexs_env/bin/python3"
LOG_DIR="/root/synexs/logs"

cd "$SYNEXS_DIR"
mkdir -p "$LOG_DIR"

# Wait for network and Redis
sleep 10

# Kill any existing instances
pkill -f "listener.py" 2>/dev/null
pkill -f "honeypot_server.py" 2>/dev/null
pkill -f "ai_swarm_fixed.py" 2>/dev/null

# Start listener.py
echo "Starting listener.py..."
nohup $PYTHON listener.py >> "$LOG_DIR/listener_output.log" 2>&1 &
LISTENER_PID=$!

# Start honeypot_server.py (if not running)
if ! pgrep -f "honeypot_server.py" > /dev/null; then
    echo "Starting honeypot_server.py..."
    nohup $PYTHON honeypot_server.py >> "$LOG_DIR/honeypot_output.log" 2>&1 &
    HONEYPOT_PID=$!
fi

# Start ai_swarm_fixed.py (if not running)
if ! pgrep -f "ai_swarm_fixed.py" > /dev/null; then
    echo "Starting ai_swarm_fixed.py..."
    nohup $PYTHON ai_swarm_fixed.py >> "$LOG_DIR/ai_swarm_output.log" 2>&1 &
    SWARM_PID=$!
fi

# Wait a moment for processes to start
sleep 2

# Verify processes started
echo "=== Synexs Process Status ==="
pgrep -af "listener.py|honeypot_server.py|ai_swarm_fixed.py" || echo "WARNING: Some processes may not have started"
echo "============================="
