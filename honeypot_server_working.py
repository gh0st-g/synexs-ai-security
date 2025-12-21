#!/usr/bin/env python3
"""
SYNEXS HONEYPOT - IMPROVED VERSION
Hybrid WAF + AI Detection (Simplified & Reliable)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from typing import Dict, List
import sys
import logging

app = Flask(__name__)

# Configuration
WORK_DIR = Path("/root/synexs")
HONEYPOT_LOG = WORK_DIR / "datasets" / "honeypot" / "attacks.json"
HONEYPOT_LOG.parent.mkdir(parents=True, exist_ok=True)

# Attack patterns (WAF layer)
SQL_PATTERNS = ["'", "OR 1=1", "UNION SELECT", "DROP TABLE", "--", ";--", "/**/"]
XSS_PATTERNS = ["<script>", "javascript:", "onerror=", "onload=", "alert(", "document.cookie"]
PATH_TRAVERSAL = ["../", "..\\", "%2e%2e", "....//", "..;/"]
CMD_INJECTION = ["|", "&&", ";", "`", "$(",  "$()", "${"]

# Rate limiting
RATE_LIMIT: Dict[str, List[float]] = {}
STATS = {
    "total_requests": 0,
    "waf_blocks": 0,
    "rate_limits": 0
}

def log_attack(attack_data: dict) -> None:
    """Log attack to honeypot log"""
    try:
        with open(HONEYPOT_LOG, "a") as f:
            f.write(json.dumps(attack_data) + "\n")
    except Exception as e:
        logging.error(f"Logging error: {e}")

def check_rate_limit(ip: str, limit: int = 20, window: int = 10) -> bool:
    """Check if IP exceeds rate limit"""
    now = time.time()

    if ip not in RATE_LIMIT:
        RATE_LIMIT[ip] = []

    # Clean old entries
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if now - t < window]

    # Check limit
    if len(RATE_LIMIT[ip]) >= limit:
        return True

    RATE_LIMIT[ip].append(now)
    return False

def waf_detection(user_agent: str, path: str, data: str) -> Dict:
    """WAF-based detection"""
    score = 0.0
    detected_patterns = []

    combined = f"{user_agent} {path} {data}".lower()

    # Check SQL injection
    for pattern in SQL_PATTERNS:
        if pattern.lower() in combined:
            score += 0.3
            detected_patterns.append(f"SQL:{pattern}")

    # Check XSS
    for pattern in XSS_PATTERNS:
        if pattern.lower() in combined:
            score += 0.3
            detected_patterns.append(f"XSS:{pattern}")

    # Check path traversal
    for pattern in PATH_TRAVERSAL:
        if pattern in combined:
            score += 0.4
            detected_patterns.append(f"PATH:{pattern}")

    # Check command injection
    for pattern in CMD_INJECTION:
        if pattern in combined:
            score += 0.3
            detected_patterns.append(f"CMD:{pattern}")

    return {
        "score": min(score, 1.0),
        "patterns": detected_patterns,
        "should_block": score > 0.5
    }

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Synexs Honeypot",
        "version": "1.0-improved",
        "total_requests": STATS["total_requests"],
        "waf_blocks": STATS["waf_blocks"],
        "rate_limits": STATS["rate_limits"]
    })

@app.route('/robots.txt')
def robots():
    """Fake robots.txt to attract crawlers"""
    return """User-agent: *
Disallow: /admin
Disallow: /backup
Disallow: /config
Disallow: /private
Disallow: /.git
Disallow: /.env
"""

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'])
def catch_all(path):
    """Catch all requests and analyze them"""
    STATS["total_requests"] += 1

    try:
        ip = request.remote_addr
        method = request.method
        headers = dict(request.headers)
        user_agent = headers.get('User-Agent', '')

        # Build request data
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "ip": ip,
            "method": method,
            "path": f"/{path}",
            "headers": {k: v for k, v in headers.items() if k in ['User-Agent', 'Referer', 'Host']},
            "args": dict(request.args),
            "data": request.get_data(as_text=True)[:500]
        }

        # Check rate limit
        if check_rate_limit(ip, limit=20, window=10):
            STATS["rate_limits"] += 1
            log_attack({**request_data, "result": "rate_limited"})
            return jsonify({"error": "Rate limit exceeded"}), 429

        # WAF detection
        waf_result = waf_detection(
            user_agent,
            path,
            request_data["data"]
        )

        if waf_result["should_block"]:
            STATS["waf_blocks"] += 1
            log_attack({
                **request_data,
                "waf_score": waf_result["score"],
                "patterns": waf_result["patterns"],
                "result": "waf_blocked"
            })
            return jsonify({
                "error": "Request blocked by WAF",
                "reason": "Suspicious patterns detected"
            }), 403

        # Log the request
        log_attack({**request_data, "result": "allowed", "waf_score": waf_result["score"]})

        # Return fake response based on path
        if "admin" in path.lower():
            return jsonify({
                "message": "Admin login",
                "status": "authentication_required"
            })
        elif "api" in path.lower():
            return jsonify({
                "message": "API endpoint",
                "version": "1.0",
                "endpoints": ["/api/users", "/api/data", "/api/login"]
            })
        else:
            return jsonify({
                "message": "Request processed",
                "method": method,
                "path": f"/{path}"
            })

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("="*60)
    logging.info("🚀 SYNEXS HONEYPOT - IMPROVED VERSION")
    logging.info("="*60)
    logging.info("✅ WAF Detection: Enabled")
    logging.info("✅ Rate Limiting: 20 req/10sec per IP")
    logging.info("✅ Logging: Active")
    logging.info(f"📁 Log file: {HONEYPOT_LOG}")
    logging.info("="*60)

    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)