#!/bin/bash
###############################################################################
# Execute Purple Team Agents Against Honeypot
# This script runs generated attack agents to collect training data
###############################################################################

AGENT_DIR="/root/synexs/datasets/agents"
BATCH_SIZE=50
DELAY_BETWEEN_BATCHES=5

echo "🎯 Executing Purple Team Agents for Training Data Collection"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Agent Directory: $AGENT_DIR"
echo "Batch Size: $BATCH_SIZE concurrent agents"
echo "Delay Between Batches: ${DELAY_BETWEEN_BATCHES}s"
echo ""

# Get list of all agent files
cd "$AGENT_DIR"
AGENTS=(agent_*.py)
TOTAL_AGENTS=${#AGENTS[@]}

if [ $TOTAL_AGENTS -eq 0 ]; then
    echo "❌ No agents found in $AGENT_DIR"
    exit 1
fi

echo "📊 Found $TOTAL_AGENTS agents to execute"
echo ""

# Execute agents in batches
EXECUTED=0
BATCH_NUM=1

for ((i=0; i<$TOTAL_AGENTS; i+=$BATCH_SIZE)); do
    # Calculate batch range
    START=$i
    END=$((i + BATCH_SIZE))
    if [ $END -gt $TOTAL_AGENTS ]; then
        END=$TOTAL_AGENTS
    fi

    BATCH_COUNT=$((END - START))

    echo "🚀 Batch $BATCH_NUM: Executing agents $START to $((END-1)) ($BATCH_COUNT agents)..."

    # Launch agents in background
    for ((j=$START; j<$END; j++)); do
        python3 "${AGENTS[$j]}" > /dev/null 2>&1 &
    done

    # Wait for batch to complete
    wait

    EXECUTED=$((EXECUTED + BATCH_COUNT))
    echo "  ✅ Batch $BATCH_NUM complete ($EXECUTED/$TOTAL_AGENTS total)"

    # Delay between batches (except for last batch)
    if [ $END -lt $TOTAL_AGENTS ]; then
        echo "  ⏳ Waiting ${DELAY_BETWEEN_BATCHES}s before next batch..."
        sleep $DELAY_BETWEEN_BATCHES
    fi

    BATCH_NUM=$((BATCH_NUM + 1))
    echo ""
done

echo "════════════════════════════════════════════════════════════════════"
echo "✅ All agents executed: $EXECUTED agents"
echo ""
echo "📊 Training Data Logs:"
echo "  • Attack metadata: /root/synexs/datasets/logs/attacks_log.jsonl"
echo "  • Response data:   /root/synexs/datasets/logs/agent_results.jsonl"
echo ""
echo "📈 View statistics:"
echo "  • Attack types:    cat /root/synexs/datasets/logs/attacks_log.jsonl | jq -r '.attack_type' | sort | uniq -c | sort -rn"
echo "  • Response codes:  cat /root/synexs/datasets/logs/agent_results.jsonl | jq -r '.status' | sort | uniq -c | sort -rn"
echo "  • Total requests:  wc -l /root/synexs/datasets/logs/agent_results.jsonl"
echo ""
echo "════════════════════════════════════════════════════════════════════"
