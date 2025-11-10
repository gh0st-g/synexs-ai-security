# 🕷️ Crawler Impersonation Detection Training

## Overview

Enhanced defensive security training system that teaches the honeypot to detect **fake search engine crawler bots** - a common evasion technique used by attackers.

---

## 🎯 Purpose

Attackers often impersonate legitimate search engine crawlers (Googlebot, Bingbot, etc.) to bypass WAF rules and avoid detection. This training module helps your honeypot learn to detect these imposters.

###  Why Crawlers Get Special Treatment:
- **Legitimate crawlers** are usually allowed through WAFs
- Blocking real crawlers hurts SEO
- Attackers exploit this by spoofing User-Agent strings
- **Detection method**: Verify IP address matches known crawler ranges

---

## 🆕 New Features

### 1. **Crawler Impersonation Profiles** (`propagate_v3.py`)

Added 5 realistic crawler profiles:

```python
CRAWLER_PROFILES = {
    'googlebot': {
        'user_agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'ip_claim': '66.249.66.1',
        'delay_range': (1.5, 5.0),
        'referer': 'https://www.google.com/',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    },
    'bingbot': {...},
    'yahoobot': {...},
    'duckduckbot': {...},
    'baiduspider': {...}
}
```

### 2. **Fake Crawler Detection** (`honeypot_server.py`)

The honeypot now:
- ✅ Detects User-Agents claiming to be crawlers
- ✅ Verifies IP matches known crawler ranges
- ✅ Blocks fake crawlers with 403 Forbidden
- ✅ Logs all detections for analysis

---

## 🔍 How It Works

### Attack Flow (Agent → Honeypot):

```
1. Agent selects "crawler_impersonate" strategy
2. Picks random crawler (e.g., Googlebot)
3. Builds realistic headers:
   - User-Agent: Googlebot string
   - Referer: google.com
   - Accept: text/html...
4. Visits multiple pages with realistic delays (1.5-5s)
5. Honeypot checks: "Is this really Googlebot?"
6. IP comparison: 127.0.0.1 != 66.249.66.1
7. Result: FAKE CRAWLER DETECTED → 403 Blocked
8. Agent reports: "crawler_impersonate | googlebot | 5 pages | 100% blocked"
```

### Detection Logic:

```python
def detect_fake_crawler(user_agent, ip):
    # 1. Check if User-Agent claims to be a crawler
    if "googlebot" in user_agent.lower():
        # 2. Verify IP matches legitimate range
        if not ip.startswith("66.249."):
            return {"is_fake": True, "reason": "IP mismatch"}
    return {"is_fake": False}
```

---

## 📊 Attack Strategy Details

### **Strategy 6: Crawler Impersonation**

**Behavior:**
- Mimics real crawler patterns
- Visits 5 pages: `/`, `/robots.txt`, `/sitemap.xml`, `/about`, `/contact`
- Uses realistic delays (1.5-7s between requests)
- Sets proper HTTP headers

**Logged Data:**
```json
{
  "type": "crawler_impersonate",
  "crawler": "googlebot",
  "page": "/robots.txt",
  "status": 403,
  "blocked": true,
  "ip_mismatch": true,
  "delay": "2.34s"
}
```

---

## 🏗️ Implementation Details

### Honeypot Enhancements:

#### 1. Crawler Pattern Database
```python
CRAWLER_PATTERNS = {
    'googlebot': ['googlebot', 'google.com/bot'],
    'bingbot': ['bingbot', 'bing.com/bingbot'],
    'yahoobot': ['yahoo! slurp', 'yahoo.com'],
    ...
}
```

#### 2. Legitimate IP Ranges
```python
LEGITIMATE_CRAWLER_IPS = {
    'googlebot': ['66.249.', '64.233.', '66.102.', '72.14.'],
    'bingbot': ['207.46.', '157.55.', '40.77.', '65.52.'],
    ...
}
```

#### 3. Detection on Every Endpoint
All endpoints now check for fake crawlers:
- `/` - Root
- `/robots.txt` - Crawler instructions
- `/login` - Login page
- `/api/data` - API endpoint
- `/admin`, `/backup`, etc. - Honeypot directories

---

## 🚀 Usage

### Test Crawler Impersonation:

```bash
# 1. Start honeypot
python3 honeypot_server.py

# 2. Start listener
python3 listener.py &

# 3. Spawn agents (includes crawler_impersonate strategy)
python3 propagate_v3.py

# 4. Run agents
python3 datasets/agents/sx*.py
```

### Expected Output:

**Agent:**
```
[sx1762272645123456] crawler_impersonate: 5 attacks | 5 blocked | 100.0% block rate
```

**Honeypot Log:**
```json
{
  "timestamp": "2025-01-04T15:30:00",
  "ip": "127.0.0.1",
  "endpoint": "/robots.txt",
  "fake_crawler": {
    "is_fake": true,
    "crawler_type": "googlebot",
    "claimed_crawler": "googlebot",
    "actual_ip": "127.0.0.1",
    "reason": "IP mismatch - not from legitimate crawler IP range"
  },
  "user_agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
}
```

### View Statistics:

```bash
curl http://127.0.0.1:8080/stats
```

```json
{
  "total_attempts": 1247,
  "blocked": 612,
  "allowed": 635,
  "fake_crawlers_detected": 156,
  "block_rate": "49.1%"
}
```

---

## 🎓 What The System Learns

### Detection Patterns:

1. **User-Agent Analysis**
   - Recognizes 5 major crawler signatures
   - Flags mismatches immediately

2. **IP Verification**
   - Compares source IP to known ranges
   - Googlebot from 192.168.x.x? → FAKE!

3. **Behavior Analysis**
   - Crawlers visit /robots.txt first
   - Use specific Accept headers
   - Follow predictable patterns

4. **Block Rate Optimization**
   - Currently ~100% detection for fake crawlers
   - Real crawlers (if any) would pass through

---

## 📈 Training Insights

### Example Training Cycle:

```
Round 1: Agent impersonates Googlebot
- Visits /robots.txt with Googlebot UA
- IP: 127.0.0.1 (localhost)
- Honeypot: "Googlebot IPs start with 66.249.x.x"
- Result: BLOCKED ❌

Round 2: System learns pattern
- Fake crawler detection rule strengthened
- Added to attack type statistics

Round 3: More variants tested
- Bingbot, Yahoo, DuckDuckGo
- All detected via IP mismatch
```

---

## 🔬 Real-World Scenarios

### Scenario 1: Attacker Spoofs Googlebot
```
Attacker → Sends SQLi with Googlebot UA
Honeypot → Checks IP: 203.0.113.45
Honeypot → Not in Googlebot range (66.249.x.x)
Result → Blocked + Logged as fake crawler
```

### Scenario 2: Legitimate Googlebot
```
Real Googlebot → Crawls site
IP → 66.249.79.123 ✓
Honeypot → IP matches known range
Result → Allowed through
```

---

## 🛡️ Defense Recommendations

Based on training data, implement:

1. **Reverse DNS Verification**
   - `host 66.249.79.123` → `crawl-66-249-79-123.googlebot.com`
   - `host crawl-66-249-79-123.googlebot.com` → `66.249.79.123`

2. **Challenge-Response**
   - Serve CAPTCHA to suspected fake crawlers
   - Real crawlers fail CAPTCHA

3. **Rate Limiting**
   - Real crawlers respect robots.txt
   - Fake crawlers often ignore delays

---

## 📊 Statistics Tracked

New metrics in `attack_stats.json`:
```json
{
  "attack_types": {
    "crawler_impersonate": 156,
    "sqli": 245,
    "xss": 189,
    ...
  },
  "crawler_detections": {
    "googlebot": 45,
    "bingbot": 38,
    "yahoobot": 23,
    "duckduckbot": 28,
    "baiduspider": 22
  }
}
```

---

## 🎯 Training Objectives

- [x] Detect 5 major crawler types
- [x] Verify IP ranges for each
- [x] Log all fake crawler attempts
- [x] Realistic timing (1.5-7s delays)
- [x] Proper HTTP headers
- [x] Multi-page crawling behavior

---

## 🚦 Next Steps

1. **Add More Crawlers**
   - Yandex, Seznam, Sogou
   - Social media crawlers (Facebook, Twitter)

2. **Advanced Detection**
   - TLS fingerprinting
   - HTTP/2 analysis
   - JavaScript challenges

3. **Adaptive Evasion**
   - Agents learn which crawlers work best
   - Rotate profiles based on success rate

---

## ⚖️ Ethical Use

**IMPORTANT:**
- ✅ Train honeypot detection (localhost only)
- ✅ Study crawler behavior patterns
- ✅ Improve WAF rules
- ❌ Attack real websites
- ❌ Spoof crawlers maliciously
- ❌ Bypass security without authorization

---

## 📚 References

- [Googlebot Verification](https://developers.google.com/search/docs/crawling-indexing/verifying-googlebot)
- [Bingbot IP Ranges](https://www.bing.com/toolbox/bingbot.json)
- [User-Agent Spoofing Detection](https://owasp.org/www-community/attacks/User_Agent_Spoofing)

---

**Status**: ✅ ACTIVE - Crawler impersonation detection enabled
**Version**: 2.0 - Full crawler analysis
