#!/bin/bash
# Quick deployment script for C2 server on VPS

set -e

VPS_IP="157.245.3.180"
VPS_USER="root"
C2_PORT="8080"

echo "=========================================="
echo "  Synexs C2 Server Deployment"
echo "=========================================="
echo ""

# Check if report_server.py exists
if [ ! -f "report_server.py" ]; then
    echo "❌ report_server.py not found"
    exit 1
fi

# Check if we can reach VPS
echo "🔍 Testing VPS connection..."
if ping -c 1 "$VPS_IP" &> /dev/null; then
    echo "✅ VPS reachable"
else
    echo "⚠️  VPS not reachable (may still work via SSH)"
fi

# Create datasets directory on VPS
echo ""
echo "📁 Creating datasets directory on VPS..."
ssh "$VPS_USER@$VPS_IP" "mkdir -p /root/synexs/datasets"

# Upload report server
echo "📤 Uploading report_server.py..."
scp report_server.py "$VPS_USER@$VPS_IP:/root/synexs/"
chmod +x report_server.py

# Check if already running
echo ""
echo "🔍 Checking if server already running..."
RUNNING=$(ssh "$VPS_USER@$VPS_IP" "pgrep -f report_server.py || echo 'none'")

if [ "$RUNNING" != "none" ]; then
    echo "⚠️  Server already running (PID: $RUNNING)"
    read -p "Kill existing process? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh "$VPS_USER@$VPS_IP" "pkill -9 -f report_server.py"
        echo "✅ Killed old process"
        sleep 2
    else
        echo "Keeping existing process"
        exit 0
    fi
fi

# Start server in background
echo ""
echo "🚀 Starting C2 server on VPS..."
ssh "$VPS_USER@$VPS_IP" "cd /root/synexs && nohup python3 report_server.py > report_server.log 2>&1 &"

sleep 3

# Check if server started
echo ""
echo "🔍 Verifying server status..."
NEW_PID=$(ssh "$VPS_USER@$VPS_IP" "pgrep -f report_server.py || echo 'none'")

if [ "$NEW_PID" != "none" ]; then
    echo "✅ Server running (PID: $NEW_PID)"
else
    echo "❌ Server failed to start"
    echo "Checking logs..."
    ssh "$VPS_USER@$VPS_IP" "tail -20 /root/synexs/report_server.log"
    exit 1
fi

# Test endpoint
echo ""
echo "🧪 Testing health endpoint..."
sleep 2

HEALTH=$(curl -s "http://$VPS_IP:$C2_PORT/health" | grep -o "healthy" || echo "failed")

if [ "$HEALTH" = "healthy" ]; then
    echo "✅ Server is healthy!"
else
    echo "⚠️  Health check failed (may need firewall rule)"
fi

echo ""
echo "=========================================="
echo "  ✅ DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "📊 Endpoints:"
echo "  Dashboard:  http://$VPS_IP:$C2_PORT/"
echo "  Report:     http://$VPS_IP:$C2_PORT/report"
echo "  Stats:      http://$VPS_IP:$C2_PORT/stats"
echo "  Health:     http://$VPS_IP:$C2_PORT/health"
echo ""
echo "📝 Logs:"
echo "  ssh $VPS_USER@$VPS_IP 'tail -f /root/synexs/report_server.log'"
echo ""
echo "🔥 Kill server:"
echo "  ssh $VPS_USER@$VPS_IP 'pkill -9 -f report_server.py'"
echo ""
echo "🧪 Test payload:"
echo "  python3 payload_agent.py"
echo ""
