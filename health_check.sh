#!/bin/bash
echo "🔍 Synexs System Health Check"
echo "=============================="
echo ""

# Check processes
echo "📊 Running Processes:"
pgrep -f honeypot_server.py > /dev/null && echo "  ✅ Honeypot" || echo "  ❌ Honeypot (stopped)"
pgrep -f listener.py > /dev/null && echo "  ✅ Listener" || echo "  ❌ Listener (stopped)"
pgrep -f ai_swarm_fixed.py > /dev/null && echo "  ✅ AI Swarm" || echo "  ❌ AI Swarm (stopped)"
pgrep -f synexs_core_orchestrator.py > /dev/null && echo "  ✅ Orchestrator" || echo "  ❌ Orchestrator (stopped)"
echo ""

# Check data collection
echo "📈 Training Data:"
echo "  Attack Logs: $(wc -l < datasets/logs/attacks_log.jsonl 2>/dev/null || echo 0) entries"
echo "  AI Decisions: $(wc -l < ai_decisions_log.jsonl 2>/dev/null || echo 0) entries"
echo "  Agent Scripts: $(ls datasets/agents/ 2>/dev/null | wc -l) scripts"
echo ""

# Check disk usage
echo "💾 Data Size:"
du -sh datasets/ 2>/dev/null | awk '{print "  Total datasets: " $1}'
du -sh ai_decisions_log.jsonl 2>/dev/null | awk '{print "  AI decisions: " $1}'
echo ""

# Check recent orchestrator activity
echo "🕐 Recent Orchestrator Activity:"
if [ -f synexs_core.log ]; then
    tail -3 synexs_core.log | sed 's/^/  /'
else
    echo "  No orchestrator log found"
fi
echo ""

# Check configuration
echo "⚙️  Configuration Files:"
[ -f ai_config.json ] && echo "  ✅ ai_config.json" || echo "  ❌ ai_config.json missing"
[ -f attack_profiles.json ] && echo "  ✅ attack_profiles.json" || echo "  ❌ attack_profiles.json missing"
[ -f synexs_model.py ] && echo "  ✅ synexs_model.py" || echo "  ❌ synexs_model.py missing"
echo ""

echo "=============================="
echo "Run: ./start_biological_organism.sh to start all services"
