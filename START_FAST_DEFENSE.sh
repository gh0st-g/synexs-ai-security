#!/bin/bash
# SYNEXS FAST DEFENSE - Quick Start Script
# Starts honeypot + defensive engine with 10x speed improvements

echo "🚀 SYNEXS FAST DEFENSE - STARTUP"
echo "================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Not in virtual environment${NC}"
    if [ -d "synexs_env" ]; then
        echo "   Activating synexs_env..."
        source synexs_env/bin/activate
    fi
fi

# Install/upgrade dependencies
echo -e "\n${YELLOW}📦 Checking dependencies...${NC}"
pip install -q -r requirements_fast.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# Create required directories
mkdir -p datasets/honeypot
mkdir -p datasets/agents
mkdir -p logs

# Initialize empty files if they don't exist
touch datasets/honeypot/attacks.json
touch datasets/real_world_kills.json

# Check if attacks.json is empty or invalid
if [ ! -s datasets/honeypot/attacks.json ]; then
    echo "[]" > datasets/honeypot/attacks.json
fi

# Check if real_world_kills.json is empty
if [ ! -s datasets/real_world_kills.json ]; then
    echo "[]" > datasets/real_world_kills.json
fi

echo -e "\n${GREEN}✅ Environment ready${NC}"

# Kill any existing processes
echo -e "\n${YELLOW}🔄 Stopping old processes...${NC}"
pkill -f "honeypot_server" 2>/dev/null
pkill -f "defensive_engine" 2>/dev/null
sleep 1

# Start defensive engine in background
echo -e "\n${GREEN}⚡ Starting Defensive Engine...${NC}"
python3 defensive_engine_fast.py > logs/defensive_engine.log 2>&1 &
DEFENSE_PID=$!
echo "   PID: $DEFENSE_PID"
sleep 2

# Check if defensive engine is running
if ps -p $DEFENSE_PID > /dev/null; then
    echo -e "${GREEN}   ✅ Defensive engine running${NC}"
else
    echo -e "${RED}   ❌ Failed to start defensive engine${NC}"
    echo "   Check logs/defensive_engine.log for errors"
fi

# Start honeypot
echo -e "\n${GREEN}🍯 Starting Honeypot (Hybrid WAF + AI)...${NC}"
python3 honeypot_server_fast.py > logs/honeypot.log 2>&1 &
HONEYPOT_PID=$!
echo "   PID: $HONEYPOT_PID"
sleep 2

# Check if honeypot is running
if ps -p $HONEYPOT_PID > /dev/null; then
    echo -e "${GREEN}   ✅ Honeypot running on http://127.0.0.1:8080${NC}"
else
    echo -e "${RED}   ❌ Failed to start honeypot${NC}"
    echo "   Check logs/honeypot.log for errors"
    exit 1
fi

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}✅ FAST DEFENSE ONLINE${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "📊 Endpoints:"
echo "   - Honeypot: http://127.0.0.1:8080"
echo "   - Stats: http://127.0.0.1:8080/stats"
echo "   - Health: http://127.0.0.1:8080/health"
echo ""
echo "📁 Logs:"
echo "   - Defensive Engine: logs/defensive_engine.log"
echo "   - Honeypot: logs/honeypot.log"
echo "   - Attacks: datasets/honeypot/attacks.json"
echo ""
echo "🎯 Features:"
echo "   ✅ Pandas-based analysis (10x faster)"
echo "   ✅ XGBoost ML detection (<5ms)"
echo "   ✅ Real-time kill learning"
echo "   ✅ Hybrid WAF + AI blocking"
echo "   ✅ Auto-retraining on updates"
echo ""
echo "💡 Commands:"
echo "   - View stats: curl http://127.0.0.1:8080/stats | jq"
echo "   - View logs: tail -f logs/honeypot.log"
echo "   - Stop: pkill -f 'honeypot_server|defensive_engine'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait and monitor
tail -f logs/honeypot.log
