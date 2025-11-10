# 🛡️ Synexs Defensive Security Training System

## Overview

A **local-only** defensive security training system where AI agents learn by attacking a honeypot. All traffic stays on localhost (127.0.0.1) - NO external attacks.

## 🎯 Purpose

- **Defensive learning**: Agents probe vulnerabilities in controlled environment
- **Pattern recognition**: Learn what triggers WAFs, rate limits, blocks
- **Survival training**: Agents adapt based on success/failure
- **Security research**: Study attack patterns and defense mechanisms

---

## 🏗️ Architecture

```
┌──────────────────┐
│  Honeypot Server │  ← Simulated vulnerable app
│  127.0.0.1:8080  │     (Flask, fake login, APIs)
└────────┬─────────┘
         │
         │ HTTP Attacks
         │
┌────────▼─────────┐
│  Training Agents │  ← Attack honeypot only
│  (20-100 agents) │     (SQLi, XSS, dir scan)
└────────┬─────────┘
         │
         │ Report Results
         │
┌────────▼─────────┐
│    Listener      │  ← Collects attack logs
│  127.0.0.1:8443  │     (Tracks survival)
└────────┬─────────┘
         │
         │ Analytics
         │
┌────────▼─────────┐
│   Swarm AI       │  ← Analyzes patterns
│  ai_swarm_fixed  │     (Mutates strategies)
└──────────────────┘
```

---

## 📦 Components

### 1. **honeypot_server.py**
Simulated vulnerable web application
- Fake login with SQLi detection
- Directory honeypots (/admin, /backup)
- Simulated WAF (Cloudflare, AWS)
- Rate limiting (5 req/10sec)
- Attack pattern detection

**Endpoints:**
- `GET  /` - Info
- `GET  /robots.txt` - Honeypot directories
- `POST /login` - Fake login (logs SQLi)
- `GET  /api/data` - Fake API
- `GET  /admin` - Honeypot directory
- `GET  /stats` - Attack statistics

### 2. **propagate_v3.py**
Spawns defensive training agents
- 20 agents per spawn
- 5 attack strategies:
  - `sqli_probe` - SQL injection tests
  - `xss_probe` - XSS payloads
  - `dir_scan` - Directory enumeration
  - `api_fuzz` - API fuzzing
  - `rate_test` - Rate limit testing

**Security:** All targets hardcoded to `127.0.0.1:8080`

### 3. **listener.py**
Receives attack logs from agents
- Tracks agent survival
- Counts blocks/allows
- Detects dead agents (>5min silence)
- Saves statistics every 10 seconds

**Stats tracked:**
- Total attacks
- Block rate
- Attack types
- Active/dead agents

### 4. **ai_swarm_fixed.py**
AI orchestrator with defensive analytics
- Analyzes training results
- Recommends strategy mutations
- Auto-spawns agents
- Telegram notifications

**Mutations:**
- Block rate >70% → Stealth mode
- Block rate >50% → Rotate User-Agent
- Block rate <30% → Increase complexity
- High death rate → Improve error handling

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install flask requests anthropic
```

### Step 2: Start Honeypot
```bash
# Terminal 1
python3 honeypot_server.py
```

Output:
```
🍯 Synexs Honeypot Server - DEFENSIVE TRAINING
⚠️  WARNING: Simulated vulnerable endpoints
📍 Binding to: 127.0.0.1:8080 (LOCAL ONLY)
✅ Server starting...
```

### Step 3: Start Listener
```bash
# Terminal 2
python3 listener.py &
```

Output:
```
🎧 Synexs Listener - DEFENSIVE TRAINING
📍 Listening on: 127.0.0.1:8443
✅ Listener started
```

### Step 4: Spawn Training Agents
```bash
# Terminal 3
python3 propagate_v3.py
```

Output:
```
🛡️  Synexs DEFENSIVE SECURITY TRAINING
⚠️  NOTICE: Agents target LOCAL honeypot ONLY
📍 Target: 127.0.0.1:8080 (localhost)
🚀 Spawning 20 defensive agents...
  ✅ sx1762272588930235 → sqli_probe
  ✅ sx1762272588927108 → dir_scan
  ...
✅ Spawned: 20
⏱️  Duration: 0.02s
```

### Step 5: Run Agents
```bash
# Run all agents (they attack honeypot)
python3 datasets/agents/sx*.py
```

Output (per agent):
```
[sx1762272588930235] sqli_probe: 3 attacks | 2 blocked | 66.7% block rate
```

### Step 6: Monitor Results

**View honeypot stats:**
```bash
curl http://127.0.0.1:8080/stats
```

**View listener stats:**
```bash
cat datasets/agents/attack_stats.json
```

**View attack logs:**
```bash
cat datasets/agents/agent_log.json
```

---

## 📊 Example Session

```bash
# Start system
python3 honeypot_server.py &
python3 listener.py &

# Spawn and run agents
python3 propagate_v3.py
python3 datasets/agents/sx*.py

# View results
curl http://127.0.0.1:8080/stats
```

**Expected output:**
```json
{
  "total_attempts": 847,
  "blocked": 312,
  "allowed": 535,
  "rate_limited": 89,
  "attack_patterns": {
    "sqli": 245,
    "xss": 189,
    "path_traversal": 78
  },
  "block_rate": "36.8%"
}
```

---

## 🔬 Training Analysis

The swarm analyzes results and adapts:

### Example Analysis:
```
🛡️  Training: 847 attacks | 312 blocked (36.8%) | 535 allowed
💡 Medium blocks → Rotate User-Agent
```

### Mutations Applied:
- **High block rate (>70%)**: Switch to stealth mode
- **Medium blocks (>50%)**: Rotate User-Agent headers
- **Low blocks (<30%)**: Increase attack complexity
- **High death rate**: Improve error handling

---

## 🎯 Attack Strategies

### 1. **SQL Injection Probe** (`sqli_probe`)
- Payloads: `' OR 1=1--`, `admin' --`, `' UNION SELECT NULL--`
- Target: `/login` endpoint
- Detection: WAF blocks, 403 errors

### 2. **XSS Probe** (`xss_probe`)
- Payloads: `<script>alert(1)</script>`, `javascript:alert(1)`
- Target: `/login` endpoint
- Detection: Input sanitization, 403 errors

### 3. **Directory Scan** (`dir_scan`)
- Paths: `/admin`, `/backup`, `/config`, `/private`, `/secret`
- Detection: Always blocked (403), logged

### 4. **API Fuzzing** (`api_fuzz`)
- Payloads: SQLi, path traversal, XSS
- Target: `/api/data?q=payload`
- Detection: WAF blocks, rate limits

### 5. **Rate Limit Test** (`rate_test`)
- 10 rapid requests to `/`
- Detection: 429 Too Many Requests after 5th request

---

## 📈 Statistics

### Honeypot Stats (`/stats` endpoint):
```json
{
  "total_attempts": 1247,
  "blocked": 498,
  "allowed": 749,
  "rate_limited": 124,
  "block_rate": "39.9%",
  "attack_patterns": {
    "sqli": 312,
    "xss": 198,
    "dir_scan": 245,
    "api_fuzz": 289,
    "rate_test": 203
  }
}
```

### Listener Stats (`attack_stats.json`):
```json
{
  "total_agents": 20,
  "total_attacks": 1247,
  "total_blocked": 498,
  "total_allowed": 749,
  "active_agents": 18,
  "dead_agents": 2,
  "attack_types": {
    "sqli": 312,
    "xss": 198,
    "dir_scan": 245
  }
}
```

---

## 🔒 Security Guarantees

### ✅ Safe:
- All traffic to `127.0.0.1` (localhost)
- Honeypot binds to `127.0.0.1:8080`
- Listener binds to `127.0.0.1:8443`
- No external network access
- All targets hardcoded in agent code

### ⚠️ WARNING:
- **DO NOT** modify target IPs
- **DO NOT** expose ports publicly
- **DO NOT** use on external systems
- **FOR EDUCATIONAL USE ONLY**

---

## 🐳 Docker Deployment

```bash
# Build custom image with Flask
docker run -d --name synexs-honeypot \
  --network host \
  python:3.11-slim \
  bash -c "pip install flask && python3 honeypot_server.py"

docker run -d --name synexs-listener \
  --network host \
  -v /root/synexs:/app \
  -w /app \
  python:3.11-slim \
  python3 listener.py
```

---

## 📝 Logs

### Attack Log (`agent_log.json`):
```json
{
  "agent_id": "sx1762272588930235",
  "decision": "sqli_probe",
  "timestamp": 1762272589,
  "attacks": [
    {
      "type": "sqli",
      "payload": "' OR 1=1--",
      "status": 403,
      "blocked": true
    }
  ],
  "stats": {
    "total": 3,
    "blocked": 2,
    "allowed": 1,
    "block_rate": "66.7%"
  }
}
```

### Honeypot Log (`attacks.json`):
```json
{
  "timestamp": "2025-01-04T12:34:56",
  "ip": "127.0.0.1",
  "endpoint": "/login",
  "method": "POST",
  "patterns": {"sqli": true},
  "result": "waf_blocked",
  "waf": "cloudflare_sim"
}
```

---

## 🎓 Educational Use Cases

1. **Security Training**: Learn attack patterns
2. **WAF Testing**: Test detection rules
3. **Rate Limiting**: Study throttling mechanisms
4. **Agent Survival**: ML for evasion (defensive)
5. **Pattern Recognition**: Analyze attack signatures

---

## 🚦 Troubleshooting

### Honeypot not starting:
```bash
# Check port 8080
lsof -i :8080
# Kill existing process
kill $(lsof -t -i:8080)
```

### Listener not receiving data:
```bash
# Check listener is running
ps aux | grep listener.py
# Check port 8443
lsof -i :8443
```

### Agents not attacking:
```bash
# Check honeypot is up
curl http://127.0.0.1:8080
# Check agent logs
cat datasets/agents/agent_log.json
```

---

## 📊 Performance

- **Agent spawn**: <0.1s for 20 agents
- **Attack execution**: 3-10 attacks per agent
- **Honeypot response**: <10ms per request
- **Listener throughput**: 100+ concurrent connections

---

## 🎯 Next Steps

1. Add more attack strategies (CSRF, XXE, SSRF)
2. Implement adaptive User-Agent rotation
3. Add stealth mode (slow requests, random delays)
4. Machine learning for optimal attack patterns
5. Multi-honeypot support

---

## ⚖️ Legal & Ethics

**IMPORTANT:**
- This system is for **EDUCATIONAL USE ONLY**
- All traffic is **LOCAL (127.0.0.1)**
- **NO external attacks** allowed
- Designed for **defensive security training**
- Use responsibly and legally

**DO NOT:**
- Attack external systems
- Use without authorization
- Modify targets to external IPs
- Deploy in production

---

## 📚 References

- [OWASP Top 10](https://owasp.org/Top10/)
- [WAF Bypass Techniques](https://owasp.org/www-community/attacks/WAF_Evasion)
- [Honeypot Research](https://www.sans.org/white-papers/)

---

**Status**: ✅ PRODUCTION READY (LOCAL ONLY)
