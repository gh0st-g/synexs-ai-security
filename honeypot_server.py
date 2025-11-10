#!/usr/bin/env python3
"""
Honeypot Server - Defensive Security Training
Simulates vulnerable services for agent learning
All traffic is LOCAL only - for security research
"""

import json
import random
import time
import ipaddress
import atexit
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response
import hashlib
import logging
from typing import Dict, List
from functools import lru_cache
import dns.resolver

app = Flask(__name__)

# Configuration
HONEYPOT_LOG = "/app/datasets/honeypot/attacks.json"
Path(HONEYPOT_LOG).parent.mkdir(parents=True, exist_ok=True)

# Batch write buffer for performance
attack_buffer = []
BUFFER_SIZE = 50
BUFFER_FLUSH_INTERVAL = 10  # seconds
last_flush_time = time.time()

# Attack patterns to detect
SQL_PATTERNS = ["'", "OR 1=1", "UNION SELECT", "DROP TABLE", "--", ";--"]
XSS_PATTERNS = ["<script>", "javascript:", "onerror=", "onload="]
PATH_TRAVERSAL = ["../", "..\\", "%2e%2e", "....//"]

# Known crawler User-Agent patterns
CRAWLER_PATTERNS: Dict[str, List[str]] = {
    'googlebot': ['googlebot', 'google.com/bot'],
    'bingbot': ['bingbot', 'bing.com/bingbot'],
    'yahoobot': ['yahoo! slurp', 'yahoo.com'],
    'duckduckbot': ['duckduckbot', 'duckduckgo.com'],
    'baiduspider': ['baiduspider', 'baidu.com/search/spider']
}

# Legitimate crawler IP ranges (CIDR format for accurate validation)
CRAWLER_IP_RANGES: Dict[str, List[str]] = {
    'googlebot': ['66.249.64.0/19', '216.239.32.0/19', '64.233.160.0/19', '66.102.0.0/20', '72.14.192.0/18'],
    'bingbot': ['157.55.0.0/16', '157.56.0.0/16', '40.77.0.0/16', '207.46.0.0/16'],
    'yahoobot': ['74.6.0.0/16', '98.136.0.0/16', '67.195.0.0/16'],
    'duckduckbot': ['23.21.227.64/26', '50.16.241.96/27'],
    'baiduspider': ['220.181.108.0/24', '123.125.71.0/24']
}

# Simulated WAF/Defense rules
BLOCK_RATE = 0.4  # 40% of attacks get blocked
RATE_LIMIT: Dict[str, List[float]] = {}  # IP rate limiting

def flush_attack_buffer() -> None:
    """Flush buffered attacks to disk"""
    global attack_buffer, last_flush_time
    if not attack_buffer:
        return

    try:
        with open(HONEYPOT_LOG, "a") as f:
            for attack in attack_buffer:
                f.write(json.dumps(attack) + "\n")
        attack_buffer.clear()
        last_flush_time = time.time()
    except Exception as e:
        logging.error(f"Flush error: {e}")

def log_attack(attack_data: dict) -> None:
    """Log attack to buffer (batch writes for performance)"""
    global attack_buffer, last_flush_time
    try:
        attack_buffer.append(attack_data)

        # Flush if buffer full or time interval exceeded
        if len(attack_buffer) >= BUFFER_SIZE or (time.time() - last_flush_time) >= BUFFER_FLUSH_INTERVAL:
            flush_attack_buffer()
    except Exception as e:
        logging.error(f"Log error: {e}")

# Register cleanup on exit
atexit.register(flush_attack_buffer)

def check_rate_limit(ip: str) -> bool:
    """Check if IP is rate limited (5 requests per 10 seconds)"""
    now = time.time()
    RATE_LIMIT.setdefault(ip, [])
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if now - t < 10]
    return len(RATE_LIMIT[ip]) >= 5

def detect_attack_pattern(payload: str) -> dict:
    """Detect attack patterns in payload"""
    payload_lower = payload.lower()
    return {
        "sqli": any(p.lower() in payload_lower for p in SQL_PATTERNS),
        "xss": any(p.lower() in payload_lower for p in XSS_PATTERNS),
        "path_traversal": any(p in payload for p in PATH_TRAVERSAL),
    }

@lru_cache(maxsize=1000)
def is_ip_in_crawler_ranges(ip: str, crawler_name: str) -> bool:
    """
    Check if IP belongs to legitimate crawler CIDR ranges
    Uses ipaddress module for accurate CIDR validation
    """
    try:
        ip_obj = ipaddress.ip_address(ip)
        for cidr in CRAWLER_IP_RANGES.get(crawler_name, []):
            network = ipaddress.ip_network(cidr, strict=False)
            if ip_obj in network:
                return True
        return False
    except (ValueError, TypeError):
        return False

@lru_cache(maxsize=1000)
def validate_ptr(ip: str, crawler_name: str) -> dict:
    """
    Validate PTR (reverse DNS) record for crawler IP (cached for performance)
    Real crawlers have matching PTR records:
    - Googlebot: *.google.com or *.googlebot.com
    - Bingbot: *.search.msn.com

    Returns: {"valid": bool, "ptr": str or None, "reason": str}
    """
    try:
        answers = dns.resolver.resolve_address(ip)
        for rdata in answers:
            ptr = str(rdata).lower().rstrip('.')
            if crawler_name == 'googlebot':
                if 'google.com' in ptr or 'googlebot.com' in ptr:
                    return {"valid": True, "ptr": ptr, "reason": "Valid PTR"}
            elif crawler_name == 'bingbot':
                if 'search.msn.com' in ptr or 'msn.com' in ptr:
                    return {"valid": True, "ptr": ptr, "reason": "Valid PTR"}
            elif crawler_name == 'yahoobot':
                if 'yahoo.com' in ptr or 'crawl.yahoo.net' in ptr:
                    return {"valid": True, "ptr": ptr, "reason": "Valid PTR"}
            elif crawler_name == 'duckduckbot':
                if 'duckduckgo.com' in ptr:
                    return {"valid": True, "ptr": ptr, "reason": "Valid PTR"}
            elif crawler_name == 'baiduspider':
                if 'baidu.com' in ptr or 'baidu.jp' in ptr:
                    return {"valid": True, "ptr": ptr, "reason": "Valid PTR"}
        return {
            "valid": False,
            "ptr": ptr if 'ptr' in locals() else None,
            "reason": f"PTR '{ptr if 'ptr' in locals() else 'none'}' doesn't match {crawler_name}"
        }
    except dns.resolver.NXDOMAIN:
        return {"valid": False, "ptr": None, "reason": "No PTR record (NXDOMAIN)"}
    except dns.resolver.NoAnswer:
        return {"valid": False, "ptr": None, "reason": "No PTR answer"}
    except dns.resolver.Timeout:
        return {"valid": False, "ptr": None, "reason": "DNS timeout"}
    except Exception as e:
        logging.warning(f"PTR validation error for {ip}: {e}")
        return {"valid": False, "ptr": None, "reason": f"DNS error: {type(e).__name__}"}

def detect_fake_crawler(user_agent: str, ip: str) -> dict:
    """
    Detect fake crawler impersonation using:
    1. CIDR-based IP validation
    2. PTR (reverse DNS) validation
    Real crawlers have both matching IP ranges AND valid PTR records
    """
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

def should_block() -> bool:
    """Randomly block requests to simulate WAF"""
    return random.random() < BLOCK_RATE

@app.route('/')
def index():
    """Root endpoint"""
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    crawler_check = detect_fake_crawler(user_agent, ip)

    if crawler_check["is_fake"]:
        log_attack({
            "timestamp": datetime.now().isoformat(),
            "ip": ip,
            "endpoint": "/",
            "fake_crawler": crawler_check,
            "user_agent": user_agent
        })
        return jsonify({
            "error": "Fake crawler detected",
            "message": "User-Agent claims to be crawler but IP doesn't match"
        }), 403

    return jsonify({
        "service": "Synexs Honeypot",
        "version": "2.0",
        "status": "active"
    })

@app.route('/robots.txt')
def robots():
    """Fake robots.txt with honeypot directories"""
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    crawler_check = detect_fake_crawler(user_agent, ip)
    log_attack({
        "timestamp": datetime.now().isoformat(),
        "ip": ip,
        "endpoint": "/robots.txt",
        "crawler_check": crawler_check,
        "user_agent": user_agent
    })
    return Response("""User-agent: *
Disallow: /admin
Disallow: /backup
Disallow: /config
Disallow: /private
Disallow: /secret
""", mimetype='text/plain')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Fake login endpoint - logs all attempts"""
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', 'unknown')

    if check_rate_limit(ip):
        log_attack({
            "timestamp": datetime.now().isoformat(),
            "ip": ip,
            "endpoint": "/login",
            "result": "rate_limited"
        })
        return jsonify({"error": "Rate limit exceeded"}), 429

    try:
        payload = str(request.get_data(as_text=True)) + str(request.args)
    except UnicodeDecodeError:
        payload = ""
    patterns = detect_attack_pattern(payload)
    crawler_check = detect_fake_crawler(user_agent, ip)

    if crawler_check["is_fake"] or (patterns and should_block()):
        log_attack({
            "timestamp": datetime.now().isoformat(),
            "ip": ip,
            "endpoint": "/login",
            "patterns": patterns,
            "fake_crawler": crawler_check if crawler_check["is_fake"] else None,
            "result": "waf_blocked",
            "user_agent": user_agent
        })
        return jsonify({
            "error": "Access Denied",
            "ray_id": f"cf-{hashlib.md5(str(time.time()).encode()).hexdigest()[:16]}"
        }), 403

    log_attack({
        "timestamp": datetime.now().isoformat(),
        "ip": ip,
        "endpoint": "/login",
        "patterns": patterns,
        "result": "allowed",
        "user_agent": user_agent
    })
    return jsonify({"status": "fail", "message": "Invalid credentials"})

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    """Fake API endpoint"""
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')

    if check_rate_limit(ip):
        return jsonify({"error": "Rate limit"}), 429

    try:
        payload = str(request.args) + str(request.get_data(as_text=True))
    except UnicodeDecodeError:
        payload = ""
    patterns = detect_attack_pattern(payload)
    crawler_check = detect_fake_crawler(user_agent, ip)

    if crawler_check["is_fake"] or (patterns and should_block()):
        log_attack({
            "timestamp": datetime.now().isoformat(),
            "ip": ip,
            "endpoint": "/api/data",
            "patterns": patterns,
            "fake_crawler": crawler_check if crawler_check["is_fake"] else None,
            "result": "waf_blocked"
        })
        return jsonify({"error": "Forbidden"}), 403

    log_attack({
        "timestamp": datetime.now().isoformat(),
        "ip": ip,
        "endpoint": "/api/data",
        "result": "allowed"
    })
    return jsonify({"data": [1, 2, 3, 4, 5]})

@app.route('/admin')
@app.route('/backup')
@app.route('/config')
@app.route('/private')
@app.route('/secret')
def protected_dirs():
    """Honeypot directories"""
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    path = request.path
    crawler_check = detect_fake_crawler(user_agent, ip)

    log_attack({
        "timestamp": datetime.now().isoformat(),
        "ip": ip,
        "endpoint": path,
        "result": "directory_blocked",
        "fake_crawler": crawler_check if crawler_check["is_fake"] else None,
        "waf": "aws_waf_sim"
    })
    return jsonify({
        "error": "Forbidden",
        "waf": "AWS WAF",
        "request_id": hashlib.md5(str(time.time()).encode()).hexdigest()
    }), 403

@app.route('/stats')
def stats():
    """Show honeypot statistics"""
    try:
        with open(HONEYPOT_LOG, "