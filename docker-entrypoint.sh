#!/bin/bash
set -e

echo "============================================================"
echo "🚀 Synexs Swarm - Docker Entrypoint"
echo "============================================================"

# Function to check health endpoint
check_health() {
    for i in {1..30}; do
        if curl -f http://localhost:5000/health >/dev/null 2>&1; then
            echo "✅ Dashboard health check passed"
            return 0
        fi
        sleep 2
    done
    echo "⚠️ Dashboard health check timeout (non-fatal)"
    return 1
}

# Function to start dashboard
start_dashboard() {
    echo "📊 Starting Flask dashboard..."
    python3 synexs_flask_dashboard.py &
    DASHBOARD_PID=$!
    echo "   PID: $DASHBOARD_PID"
}

# Function to start swarm
start_swarm() {
    echo "🤖 Starting AI swarm..."
    python3 ai_swarm_fixed.py &
    SWARM_PID=$!
    echo "   PID: $SWARM_PID"
}

# Function to start honeypot
start_honeypot() {
    echo "🍯 Starting honeypot server..."
    python3 honeypot_server.py &
    HONEYPOT_PID=$!
    echo "   PID: $HONEYPOT_PID"
}

# Function to start core loop
start_core() {
    echo "🧠 Starting core loop..."
    python3 synexs_core_loop2.0.py &
    CORE_PID=$!
    echo "   PID: $CORE_PID"
}

# Function to start listener
start_listener() {
    if [ -f "listener.py" ]; then
        echo "👂 Starting listener..."
        python3 listener.py &
        LISTENER_PID=$!
        echo "   PID: $LISTENER_PID"
    fi
}

# Cleanup handler
cleanup() {
    echo ""
    echo "⛔ Shutdown signal received"

    # Kill all child processes
    if [ ! -z "$DASHBOARD_PID" ]; then
        echo "   Stopping dashboard (PID: $DASHBOARD_PID)"
        kill $DASHBOARD_PID 2>/dev/null || true
    fi

    if [ ! -z "$SWARM_PID" ]; then
        echo "   Stopping swarm (PID: $SWARM_PID)"
        kill $SWARM_PID 2>/dev/null || true
    fi

    if [ ! -z "$HONEYPOT_PID" ]; then
        echo "   Stopping honeypot (PID: $HONEYPOT_PID)"
        kill $HONEYPOT_PID 2>/dev/null || true
    fi

    if [ ! -z "$CORE_PID" ]; then
        echo "   Stopping core loop (PID: $CORE_PID)"
        kill $CORE_PID 2>/dev/null || true
    fi

    if [ ! -z "$LISTENER_PID" ]; then
        echo "   Stopping listener (PID: $LISTENER_PID)"
        kill $LISTENER_PID 2>/dev/null || true
    fi

    echo "✅ Clean shutdown complete"
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT SIGQUIT

# Parse command
MODE="${1:-full}"

echo "📍 Mode: $MODE"
echo "📁 Work dir: $(pwd)"
echo "🐍 Python: $(python3 --version)"
echo "============================================================"
echo ""

case "$MODE" in
    full)
        echo "🚀 Starting FULL STACK (dashboard + swarm + honeypot)"
        start_dashboard
        sleep 5
        start_swarm
        sleep 2
        start_honeypot
        sleep 2
        start_listener
        ;;

    dashboard)
        echo "📊 Starting DASHBOARD ONLY"
        start_dashboard
        ;;

    swarm)
        echo "🤖 Starting SWARM ONLY"
        start_swarm
        ;;

    honeypot)
        echo "🍯 Starting HONEYPOT ONLY"
        start_honeypot
        ;;

    core)
        echo "🧠 Starting CORE LOOP ONLY"
        start_core
        ;;

    *)
        echo "❌ Unknown mode: $MODE"
        echo "Available modes: full, dashboard, swarm, honeypot, core"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "✅ Startup complete"
echo "============================================================"

# Monitor processes
while true; do
    # Check if any critical process died
    if [ ! -z "$DASHBOARD_PID" ] && ! kill -0 $DASHBOARD_PID 2>/dev/null; then
        echo "⚠️ Dashboard died, restarting..."
        start_dashboard
    fi

    if [ ! -z "$SWARM_PID" ] && ! kill -0 $SWARM_PID 2>/dev/null; then
        echo "⚠️ Swarm died, restarting..."
        start_swarm
    fi

    if [ ! -z "$HONEYPOT_PID" ] && ! kill -0 $HONEYPOT_PID 2>/dev/null; then
        echo "⚠️ Honeypot died, restarting..."
        start_honeypot
    fi

    sleep 30
done
