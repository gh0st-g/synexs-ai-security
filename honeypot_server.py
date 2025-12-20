#!/usr/bin/env python3
"""
SYNEXS HONEYPOT - HYBRID WAF + AI DETECTION
10x faster with XGBoost integration + real-time learning
"""

import json
import random
import time
import ipaddress
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response
import hashlib
import logging
from typing import Dict, List, Optional
import dns.resolver
import sys
import concurrent.futures

# Import defensive engine
sys.path.append('/root/synexs')
try:
    from defensive_engine_fast import predict_block, load_xgboost_model, analyze_blocks_fast
    AI_ENABLED = True
except ImportError:
    logging.error("Defensive engine not found - running in fallback mode")
    AI_ENABLED = False

app = Flask(__name__)

# Configuration
HONEYPOT_LOG = "/root/synexs/datasets/honeypot/attacks.json"
Path(HONEYPOT_LOG).parent.mkdir(parents=True, exist_ok=True)

# Attack patterns to detect (WAF layer)
SQL_PATTERNS = ["'", "OR 1=1", "UNION SELECT", "DROP TABLE", "--", ";--", "/**/"]
XSS_PATTERNS = ["<script>", "javascript:", "onerror=", "onload=", "alert(", "document.cookie"]
PATH_TRAVERSAL = ["../", "..\\", "%2e%2e", "....//", "..;/"]
CMD_INJECTION = ["|", "&&", ";", "`", "$(",  "$()", "${"]

# Known crawler patterns
CRAWLER_PATTERNS: Dict[str, List[str]] = {
    'googlebot': ['googlebot', 'google.com/bot'],
    'bingbot': ['bingbot', 'bing.com/bingbot'],
    'yahoobot': ['yahoo! slurp', 'yahoo.com'],
    'duckduckbot': ['duckduckbot', 'duckduckgo.com'],
    'baiduspider': ['baiduspider', 'baidu.com/search/spider']
}

# Legitimate crawler IP ranges (CIDR)
CRAWLER_IP_RANGES: Dict[str, List[str]] = {
    'googlebot': ['66.249.64.0/19', '216.239.32.0/19', '64.233.160.0/19', '66.102.0.0/20', '72.14.192.0/18'],
    'bingbot': ['157.55.0.0/16', '157.56.0.0/16', '40.77.0.0/16', '207.46.0.0/16'],
    'yahoobot': ['74.6.0.0/16', '98.136.0.0/16', '67.195.0.0/16'],
    'duckduckbot': ['23.21.227.64/26', '50.16.241.96/27'],
    'baiduspider': ['220.181.108.0/24', '123.125.71.0/24']
}

# Hybrid detection settings
WAF_THRESHOLD = 0.5  # If WAF score > 0.5, block immediately
AI_THRESHOLD = 0.7   # If AI confidence > 0.7, block
RATE_LIMIT: Dict[str, List[float]] = {}
BLOCK_CACHE: Dict[str, float] = {}  # Cache blocks to prevent repeated processing

# Stats
STATS = {
    "total_requests": 0,
    "waf_blocks": 0,
    "ai_blocks": 0,
    "hybrid_blocks": 0,
    "allowed": 0,
    "avg_response_time_ms": 0.0
}

# Load AI model on startup
if AI_ENABLED:
    try:
        load_xgboost_model()
        logging.info("AI detection enabled")
    except Exception as e:
        logging.error(f"Error loading XGBoost model: {e}")
        AI_ENABLED = False

def log_attack(attack_data: dict) -> None:
    """Log attack to honeypot log (async write)"""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(write_to_log, attack_data)
    except Exception as e:
        logging.error(f"Log error: {e}")

def write_to_log(attack_data: dict) -> None:
    """Write attack data to the log file"""
    try:
        with open(HONEYPOT_LOG, "a") as f:
            f.write(json.dumps(attack_data) + "\n")
    except Exception as e:
        logging.error(f"Log error: {e}")

def check_rate_limit(ip: str, limit: int = 10, window: int = 10) -> bool:
    """
    Advanced rate limiting
    Default: 10 requests per 10 seconds
    Returns True if rate limited
    """
    now = time.time()
    RATE_LIMIT.setdefault(ip, [])
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if now - t < window]

    if len(RATE_LIMIT[ip]) >= limit:
        return True

    RATE_LIMIT[ip].append(now)
    return False

def waf_score(payload: str, user_agent: str, path: str) -> Dict:
    """
    WAF layer: Fast pattern matching
    Returns: {"score": 0.0-1.0, "threats": [], "method": "waf"}
    """
    threats = []
    score = 0.0

    payload_lower = payload.lower()
    ua_lower = user_agent.lower()

    # SQL injection detection
    if any(p.lower() in payload_lower for p in SQL_PATTERNS):
        threats.append("sqli")
        score += 0.4

    # XSS detection
    if any(p.lower() in payload_lower for p in XSS_PATTERNS):
        threats.append("xss")
        score += 0.4

    # Path traversal
    if any(p in payload for p in PATH_TRAVERSAL):
        threats.append("path_traversal")
        score += 0.3

    # Command injection
    if any(p in payload for p in CMD_INJECTION):
        threats.append("cmd_injection")
        score += 0.5

    # Suspicious paths
    if any(p in path.lower() for p in ['/admin', '/backup', '/config', '/.git', '/.env']):
        threats.append("sensitive_path")
        score += 0.2

    # Suspicious user agents
    if any(p in ua_lower for p in ['scanner', 'bot', 'curl', 'wget', 'python']):
        if not any(c in ua_lower for c in ['googlebot', 'bingbot']):
            threats.append("suspicious_ua")
            score += 0.1

    return {
        "score": min(score, 1.0),
        "threats": threats,
        "method": "waf"
    }

def is_ip_in_crawler_ranges(ip: str, crawler_name: str) -> bool:
    """Check if IP is in legitimate crawler CIDR ranges"""
    try:
        ip_obj = ipaddress.ip_address(ip)
        ranges = CRAWLER_IP_RANGES.get(crawler_name, [])
        for cidr in ranges:
            if ip_obj in ipaddress.ip_network(cidr):
                return True
        return False
    except Exception as e:
        logging.error(f"CIDR validation error: {e}")
        return False

def validate_ptr(ip: str, crawler_name: str) -> dict:
    """Validate PTR record for crawler IP"""
    try:
        reversed_dns = dns.resolver.resolve_address(ip)
        if reversed_dns:
            ptr = str(reversed_dns[0])
            # Check if PTR matches expected crawler domain
            expected_domains = CRAWLER_PATTERNS.get(crawler_name, [])
            for domain in expected_domains:
                if domain.replace('/', '') in ptr.lower():
                    return {"valid": True, "ptr": ptr}
            return {"valid": False, "ptr": ptr, "reason": f"PTR {ptr} doesn't match expected domain"}
        return {"valid": False, "ptr": None, "reason": "No PTR record"}
    except Exception as e:
        return {"valid": False, "ptr": None, "reason": f"DNS lookup failed: {str(e)}"}

def detect_fake_crawler(user_agent: str, ip: str) -> dict:
    """Detect fake crawler impersonation (CIDR + PTR validation)"""
    if not user_agent:
        return {"is_fake": False, "crawler_type": None}

    ua_lower = user_agent.lower()

    for crawler_name, patterns in CRAWLER_PATTERNS.items():
        if any(pattern in ua_lower for pattern in patterns):
            is_legitimate = is_ip_in_crawler_ranges(ip, crawler_name)
            if not is_legitimate:
                return {
                    "is_fake": True,
                    "crawler_type": crawler_name,
                    "claimed_crawler": crawler_name,
                    "actual_ip": ip,
                    "reason": f"IP {ip} not in legitimate {crawler_name} CIDR ranges",
                    "validation_failed": "CIDR"
                }

            ptr_result = validate_ptr(ip, crawler_name)
            if not ptr_result["valid"]:
                return {
                    "is_fake": True,
                    "crawler_type": crawler_name,
                    "claimed_crawler": crawler_name,
                    "actual_ip": ip,
                    "ptr": ptr_result["ptr"],
                    "reason": f"PTR validation failed: {ptr_result['reason']}",
                    "validation_failed": "PTR"
                }

            return {
                "is_fake": False,
                "crawler_type": crawler_name,
                "validated": True,
                "ptr": ptr_result["ptr"],
                "validation_passed": ["CIDR", "PTR"]
            }

    return {"is_fake": False, "crawler_type": None}

def hybrid_detection(user_agent: str, path: str, ip: str, payload: str = "") -> Dict:
    """
    HYBRID DETECTION: WAF + AI
    1. Fast WAF check (pattern matching)
    2. XGBoost AI prediction if WAF uncertain
    Returns: {"should_block": bool, "confidence": float, "method": str, "threats": []}
    """
    start = time.time()

    # Check cache (prevent repeated analysis of same IP)
    cache_key = f"{ip}:{user_agent}:{path}"
    if cache_key in BLOCK_CACHE:
        if time.time() - BLOCK_CACHE[cache_key] < 60:  # Cache for 60s
            return {
                "should_block": True,
                "confidence": 1.0,
                "method": "cache",
                "threats": ["cached_block"],
                "latency_ms": 0
            }

    # Layer 1: WAF (fast pattern matching)
    waf_result = waf_score(payload, user_agent, path)

    if waf_result["score"] >= WAF_THRESHOLD:
        # High WAF score - block immediately
        STATS["waf_blocks"] += 1
        BLOCK_CACHE[cache_key] = time.time()
        return {
            "should_block": True,
            "confidence": waf_result["score"],
            "method": "waf",
            "threats": waf_result["threats"],
            "latency_ms": (time.time() - start) * 1000
        }

    # Layer 2: AI prediction (if WAF uncertain)
    if AI_ENABLED and 0.2 <= waf_result["score"] < WAF_THRESHOLD:
        try:
            ai_result = predict_block(user_agent, path, ip)
        except Exception as e:
            logging.error(f"AI prediction error: {e}")
            ai_result = {"should_block": False, "confidence": 0.0}

        if ai_result["should_block"] and ai_result["confidence"] >= AI_THRESHOLD:
            STATS["ai_blocks"] += 1
            BLOCK_CACHE[cache_key] = time.time()
            return {
                "should_block": True,
                "confidence": ai_result["confidence"],
                "method": "ai",
                "threats": waf_result["threats"] + ["ai_detected"],
                "latency_ms": (time.time() - start) * 1000
            }

        # Hybrid decision: WAF + AI consensus
        if waf_result["score"] > 0.3 and ai_result["confidence"] > 0.5:
            STATS["hybrid_blocks"] += 1
            BLOCK_CACHE[cache_key] = time.time()
            return {
                "should_block": True,
                "confidence": (waf_result["score"] + ai_result["confidence"]) / 2,
                "method": "hybrid",
                "threats": waf_result["threats"] + ["ai_consensus"],
                "latency_ms": (time.time() - start) * 1000
            }

    # Allow request
    STATS["allowed"] += 1
    return {
        "should_block": False,
        "confidence": waf_result["score"],
        "method": "allowed",
        "threats": waf_result["threats"],
        "latency_ms": (time.time() - start) * 1000
    }

@app.before_request
def track_request():
    """Track request start time"""
    request.start_time = time.time()
    STATS["total_requests"] += 1

@app.after_request
def update_stats(response):
    """Update response time stats"""
    if hasattr(request, 'start_time'):
        elapsed = (time.time() - request.start_time) * 1000
        # Rolling average
        alpha = 0.1  # Smoothing factor
        if STATS["avg_response_time_ms"] == 0:
            STATS["avg_response_time_ms"] = elapsed
        else:
            STATS["avg_response_time_ms"] = (alpha * elapsed) + ((1 - alpha) * STATS["avg_response_time_ms"])
    return response

@app.route('/stats')
def stats():
    """Return honeypot statistics"""
    return jsonify({
        "stats": STATS,
        "ai_enabled": AI_ENABLED,
        "uptime": time.time()
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "ai_enabled": AI_ENABLED,
        "total_requests": STATS["total_requests"]
    })

# Catch-all route for everything else
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'])
def catch_all(path):
    """Catch all requests and analyze them"""
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
        "headers": headers,
        "args": dict(request.args),
        "form": dict(request.form) if request.form else {},
        "data": request.get_data(as_text=True)[:1000]  # Limit data size
    }

    # Analyze with hybrid WAF + AI
    hybrid_result = hybrid_detection(user_agent, path, ip, request_data["data"])

    if hybrid_result["should_block"]:
        if hybrid_result["method"] ==