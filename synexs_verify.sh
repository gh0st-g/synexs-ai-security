#!/bin/bash
# SYNEXS VERIFICATION SCRIPT – Run on VPS or via SSH
# Author: Ara (Grok) – 2025-11-16
# Purpose: Confirm listener + intel + training pipeline is LIVE

set -e  # Exit on any error

echo "=== SYNEXS VERIFICATION START ==="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo

# 1. Check listener process
echo "1. Checking listener.py process..."
if pgrep -f "listener.py" > /dev/null; then
    echo "   Listener RUNNING (PID: $(pgrep -f listener.py | head -1))"
else
    echo "   LISTENER NOT RUNNING – starting..."
    nohup python3 listener.py > listener_nohup.log 2>&1 &
    sleep 3
    if pgrep -f "listener.py" > /dev/null; then
        echo "   Listener STARTED"
    else
        echo "   FAILED TO START LISTENER"
        exit 1
    fi
fi

# 2. Check Redis queue
echo
echo "2. Redis queue length..."
QUEUE_LEN=$(redis-cli llen agent_tasks)
echo "   agent_tasks queue: $QUEUE_LEN messages"

# 3. Push test message with known vuln IP
TEST_IP="192.168.99.99"
CACHE_FILE="/tmp/synexs_intel_cache/$TEST_IP.json"
echo
echo "3. Creating fake CVE cache for $TEST_IP..."
mkdir -p /tmp/synexs_intel_cache
cat > "$CACHE_FILE" << EOF
{
  "ip": "$TEST_IP",
  "timestamp": $(date +%s),
  "open_ports": [22, 80],
  "vulns": ["CVE-2021-44228"],
  "nmap": {"22": {"name": "ssh", "product": "OpenSSH", "version": "7.6p1"}}
}
EOF
echo "   Cache created: $CACHE_FILE"

# 4. Push to Redis
echo
echo "4. Pushing test message..."
redis-cli lpush agent_tasks "{\"source_ip\":\"$TEST_IP\",\"action\":\"SCAN\"}" > /dev/null
echo "   Message pushed"

# 5. Wait for processing
echo
echo "5. Waiting 8 seconds for intel + training..."
sleep 8

# 6. Check training file
echo
echo "6. Checking training_binary_v3.jsonl..."
if grep -q "$TEST_IP" training_binary_v3.jsonl; then
    echo "   SUCCESS: New training sample found!"
    tail -n 1 training_binary_v3.jsonl | jq -r '.instruction,.output,.source'
else
    echo "   NO TRAINING SAMPLE – something failed"
    echo "   Last 5 lines:"
    tail -n 5 training_binary_v3.jsonl
fi

# 7. Health endpoint
echo
echo "7. Health check..."
curl -s http://localhost:8765/health | jq '.status, .intel_hits, .training_samples_added'

echo
echo "=== VERIFICATION COMPLETE ==="
echo "If you see 'SUCCESS' + intel_hits > 0 → PIPELINE IS LIVE"