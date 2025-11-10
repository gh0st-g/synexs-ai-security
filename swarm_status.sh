#!/bin/bash
# Quick swarm status checker

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              📊 SYNEXS SWARM STATUS                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check tmux
echo "🖥️  TMUX:"
if tmux has-session -t swarm 2>/dev/null; then
    echo "  ✅ Session 'swarm' exists"
    tmux list-panes -t swarm 2>/dev/null | wc -l | xargs echo "  📦 Panes:"
else
    echo "  ❌ Session 'swarm' not found"
fi
echo ""

# Check processes
echo "🤖 PROCESSES:"
pgrep -f honeypot_server >/dev/null && echo "  ✅ Honeypot Server (PID: $(pgrep -f honeypot_server))" || echo "  ❌ Honeypot Server"
pgrep -f listener.py >/dev/null && echo "  ✅ Listener (PID: $(pgrep -f listener.py))" || echo "  ❌ Listener"
pgrep -f ai_swarm_fixed >/dev/null && echo "  ✅ AI Swarm (PID: $(pgrep -f ai_swarm_fixed))" || echo "  ❌ AI Swarm"
pgrep -f synexs_core_loop >/dev/null && echo "  ✅ Core Loop (PID: $(pgrep -f synexs_core_loop))" || echo "  ❌ Core Loop"
echo ""

# Check services
echo "🌐 SERVICES:"
curl -s http://localhost:8080/ >/dev/null 2>&1 && echo "  ✅ Honeypot: http://localhost:8080" || echo "  ❌ Honeypot not responding"
curl -s http://localhost:5000/health >/dev/null 2>&1 && echo "  ✅ Dashboard: http://localhost:5000" || echo "  ❌ Dashboard not responding"
echo ""

# Disk space
echo "💾 DISK:"
df -h /app 2>/dev/null || df -h / | grep -v "^Filesystem"
echo ""

# Uptime
echo "⏱️  SYSTEM:"
uptime | sed 's/.*up /  Up: /' | sed 's/,  load.*//'
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "🎯 View logs: tail -f honeypot.log"
echo "🎯 Attach tmux: tmux attach -t swarm"
echo "═══════════════════════════════════════════════════════════════"
