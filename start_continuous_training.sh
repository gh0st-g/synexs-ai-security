#!/bin/bash
###############################################################################
# Synexs Continuous Training Data Collection
#
# This script runs continuous attack generation for training data collection
# with Telegram API notifications for monitoring
#
# Features:
# - Generates diverse attack patterns every 5 minutes
# - Executes agents against localhost honeypot
# - Monitors attack diversity and volume
# - Runs indefinitely for continuous data collection
#
# Usage:
#   ./start_continuous_training.sh
#
# To run in background:
#   nohup ./start_continuous_training.sh > continuous_training.log 2>&1 &
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  SYNEXS CONTINUOUS TRAINING DATA COLLECTION"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Working Directory: $SCRIPT_DIR"
echo "Honeypot Target: http://127.0.0.1:8080"
echo "Collection Interval: 5 minutes"
echo ""

# Check if honeypot is running
if ! curl -s http://127.0.0.1:8080/ > /dev/null 2>&1; then
    echo "⚠️  WARNING: Honeypot not responding at 127.0.0.1:8080"
    echo "   Please start honeypot: python3 honeypot_server.py &"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Configuration
PYTHON_BIN="$SCRIPT_DIR/synexs_env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

BATCH_SIZE=50
COLLECTION_INTERVAL=300  # 5 minutes
CYCLE_COUNT=0
TOTAL_ATTACKS=0

echo "✓ Configuration loaded"
echo "  Python: $PYTHON_BIN"
echo "  Batch Size: $BATCH_SIZE agents/cycle"
echo "  Interval: $COLLECTION_INTERVAL seconds"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  STARTING CONTINUOUS COLLECTION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Continuous collection loop
while true; do
    CYCLE_COUNT=$((CYCLE_COUNT + 1))
    CYCLE_START=$(date +%s)

    echo "─────────────────────────────────────────────────────────────"
    echo "🔄 Cycle #$CYCLE_COUNT - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "─────────────────────────────────────────────────────────────"

    # Step 1: Generate diverse agents
    echo ""
    echo "📊 Step 1: Generating $BATCH_SIZE diverse attack agents with propagate_v4..."
    if $PYTHON_BIN propagate_v4.py; then
        echo "✅ Agent generation complete"
    else
        echo "❌ Agent generation failed"
        continue
    fi

    # Step 2: Count generated agents
    AGENT_COUNT=$(ls datasets/agents/agent_*.py 2>/dev/null | wc -l)
    echo "   Generated: $AGENT_COUNT agents"

    # Step 3: Execute agents
    echo ""
    echo "🚀 Step 2: Executing agents against honeypot..."
    EXECUTED=0
    FAILED=0

    for agent in datasets/agents/agent_*.py; do
        if [ -f "$agent" ]; then
            if timeout 10 $PYTHON_BIN "$agent" > /dev/null 2>&1; then
                EXECUTED=$((EXECUTED + 1))
            else
                FAILED=$((FAILED + 1))
            fi
            sleep 0.1  # Small delay between agents
        fi
    done

    echo "   Executed: $EXECUTED/$AGENT_COUNT"
    if [ $FAILED -gt 0 ]; then
        echo "   Failed: $FAILED"
    fi

    # Step 4: Clean up agents
    rm -f datasets/agents/agent_*.py

    # Step 5: Analyze diversity
    echo ""
    echo "📊 Step 3: Analyzing attack diversity..."

    # Check both honeypot logs and training logs
    if [ -f "datasets/logs/attacks_log.jsonl" ]; then
        ATTACK_COUNT=$(wc -l < datasets/logs/attacks_log.jsonl)
        echo "   Total training attacks logged: $ATTACK_COUNT"

        # Show attack distribution
        if command -v jq &> /dev/null; then
            echo ""
            echo "   Attack Type Distribution:"
            cat datasets/logs/attacks_log.jsonl | jq -r '.attack_type // "unknown"' | sort | uniq -c | sort -rn | head -10 | sed 's/^/     /'
        fi
    fi

    if [ -f "datasets/honeypot/attacks.json" ]; then
        HONEYPOT_COUNT=$(wc -l < datasets/honeypot/attacks.json)
        echo "   Honeypot detected attacks: $HONEYPOT_COUNT"
    fi

    # Calculate cycle stats
    CYCLE_END=$(date +%s)
    CYCLE_DURATION=$((CYCLE_END - CYCLE_START))
    TOTAL_ATTACKS=$((TOTAL_ATTACKS + EXECUTED))

    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo "📈 Cycle #$CYCLE_COUNT Summary:"
    echo "   Duration: ${CYCLE_DURATION}s"
    echo "   Attacks This Cycle: $EXECUTED"
    echo "   Total Attacks: $TOTAL_ATTACKS"
    echo "   Average per Cycle: $((TOTAL_ATTACKS / CYCLE_COUNT))"
    echo "─────────────────────────────────────────────────────────────"

    # Wait for next cycle
    echo ""
    echo "⏳ Waiting $COLLECTION_INTERVAL seconds until next cycle..."
    echo "   Next cycle: $(date -d "+$COLLECTION_INTERVAL seconds" '+%H:%M:%S')"
    echo ""

    sleep $COLLECTION_INTERVAL
done

# Note: This script runs indefinitely
# To stop: pkill -f start_continuous_training.sh
