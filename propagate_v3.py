#!/usr/bin/env python3
"""
propagate_v3.py - DEFENSIVE SECURITY EDITION
Agents attack LOCAL honeypot only - for security training
All targets are 127.0.0.1 (localhost) - NO external attacks
"""

import os
import json
import random
import base64
import time
import socket
import sys
import traceback
from typing import List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MEMORY_LOG = "/app/datasets/memory/memory_log.json"
AGENT_DIR = "/app/datasets/agents"
LISTENER_IP = "127.0.0.1"  # LOCAL ONLY
LISTENER_PORT = 8443
HONEYPOT_URL = "http://127.0.0.1:8080"  # LOCAL HONEYPOT ONLY
DEFAULT_AGENT_COUNT = 20
MAX_WORKERS = 10

# Ensure directories exist
Path(AGENT_DIR).mkdir(parents=True, exist_ok=True)

def load_decisions() -> List[str]:
    """Load decisions from memory log with fallback"""
    try:
        with open(MEMORY_LOG, "r") as f:
            decisions = sum([json.loads(line)["decisions"] for line in f if line.strip()], [])
            if decisions:
                return decisions
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback: Attack strategies (including crawler impersonation)
    return ["sqli_probe", "xss_probe", "dir_scan", "api_fuzz", "rate_test",
            "crawler_impersonate"] * 40

CRAWLER_PROFILES = {
    'googlebot': {
        'user_agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'ip_range': ['66.249.66.1', '66.249.66.2', '66.249.78.1', '66.249.79.1'],
        'delay_range': (1.5, 5.0),
        'referer': 'https://www.google.com/',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    },
    # ... other crawler profiles
}

# Stealth Human Profiles (for adaptive evasion when crawlers fail)
HUMAN_PROFILES = [
    {
        'name': 'Chrome 129 Win10',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'delay_range': (0.4, 1.8)
    },
    {
        'name': 'Firefox 129 macOS',
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:129.0) Gecko/20100101 Firefox/129.0',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'delay_range': (0.6, 2.2)
    },
    {
        'name': 'Safari 17 macOS',
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'delay_range': (0.5, 2.0)
    }
]

def build_scp(agent_id: str, decision: str) -> dict:
    """Build SCP signal payload"""
    return {
        "scp_version": "2.0",
        "signal_id": f"{agent_id}-{int(time.time())}",
        "origin": f"agent@defensive",
        "target": "honeypot_localhost",
        "timestamp": int(time.time()),
        "action": decision,
        "tags": [decision, "defensive_training"]
    }

def check_mutation() -> dict:
    """Check if AI swarm triggered a mutation"""
    try:
        mutation_file = Path("/app/datasets/mutation.json")
        if mutation_file.exists():
            with open(mutation_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error checking mutation: {e}")
    return {}

def generate_attack_agent_code(agent_id: str, decision: str) -> str:
    """
    Generate agent that attacks LOCAL honeypot only
    SECURITY: All targets hardcoded to 127.0.0.1
    Supports adaptive mutation to stealth human profiles
    """
    scp = build_scp(agent_id, decision)
    scp_json = json.dumps(scp, separators=(',', ':'))

    # Check for mutation
    mutation = check_mutation()
    use_stealth = mutation.get("action") == "switch_to_stealth_human"

    # Select profile based on mutation
    if decision == "crawler_impersonate" and use_stealth:
        # Use stealth human profile instead
        profile = random.choice(HUMAN_PROFILES)
        user_agent = profile['user_agent']
        accept = profile['accept']
        delay_min, delay_max = profile['delay_range']
        profile_name = profile['name']
    else:
        # Use crawler profile (original behavior)
        user_agent = CRAWLER_PROFILES['googlebot']['user_agent']
        accept = CRAWLER_PROFILES['googlebot']['accept']
        delay_min, delay_max = CRAWLER_PROFILES['googlebot']['delay_range']
        profile_name = "Googlebot"

    # Generate agent code
    return f"""# Agent {agent_id}
import requests, time, random
HONEYPOT = "{HONEYPOT_URL}"
try:
    if "{decision}" == "crawler_impersonate":
        headers = {{"User-Agent": "{user_agent}", "Accept": "{accept}"}}
        print(f"[{agent_id}] Using profile: {profile_name}")
        for page in ["/", "/robots.txt", "/api/data"]:
            resp = requests.get(HONEYPOT + page, headers=headers, timeout=5)
            print(f"[{agent_id}] {decision}: {{page}} -> {{resp.status_code}}")
            time.sleep(random.uniform({delay_min}, {delay_max}))
    else:
        resp = requests.get(HONEYPOT + "/", timeout=3)
        print(f"[{agent_id}] {decision}: -> {{resp.status_code}}")
except requests.exceptions.RequestException as e:
    print(f"[{agent_id}] Error: {{e}}")
except Exception as e:
    print(f"[{agent_id}] Unexpected error: {{e}}")
"""

def spawn_agent(agent_id: str, decision: str) -> Tuple[str, bool]:
    """Spawn a single defensive training agent"""
    try:
        code = generate_attack_agent_code(agent_id, decision)
        blob = base64.b64encode(code.encode()).decode()
        path = Path(AGENT_DIR) / f"{agent_id}.py"

        with open(path, "w") as f:
            f.write(f"# Synexs Defensive Training Agent {agent_id}\n")
            f.write(f"# TARGET: localhost honeypot ONLY - NO external attacks\n")
            f.write(f"import base64\n")
            f.write(f"exec(base64.b64decode('{blob}'))\n")

        return (agent_id, True)
    except Exception as e:
        print(f"❌ {agent_id}: {e}")
        return (agent_id, False)

def batch_spawn_agents(count: int = DEFAULT_AGENT_COUNT) -> dict:
    """Spawn defensive training agents in parallel"""
    start_time = time.time()

    # Load attack strategies
    decisions = load_decisions()
    if not decisions:
        print("❌ No strategies available")
        return {"spawned": 0, "failed": 0, "duration": 0}

    # Sample strategies
    sample = random.sample(decisions, k=count)

    # Generate agent IDs
    agent_tasks = [
        (f"sx{int(time.time()*1000) + i}{random.randint(100,999)}", decision)
        for i, decision in enumerate(sample)
    ]

    # Spawn agents in parallel
    spawned = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(spawn_agent, agent_id, decision): (agent_id, decision)
            for agent_id, decision in agent_tasks
        }

        for future in as_completed(futures):
            agent_id, success = future.result()
            if success:
                spawned += 1
                decision = futures[future][1]
                print(f"  ✅ {agent_id} → {decision}")
            else:
                failed += 1

    duration = time.time() - start_time

    return {
        "spawned": spawned,
        "failed": failed,
        "duration": duration
    }

def main():
    """Main spawner entry point"""
    print("=" * 60)
    print("🛡️  Synexs DEFENSIVE SECURITY TRAINING v2.0")
    print("=" * 60)
    print("⚠️  NOTICE: Agents target LOCAL honeypot ONLY")
    print("📍 Target: 127.0.0.1:8080 (localhost)")
    print("🎯 Purpose: Defensive security training")
    print("🕷️  NEW: Crawler impersonation detection")
    print("=" * 60)

    # Spawn agents
    print(f"\n🚀 Spawning {DEFAULT_AGENT_COUNT} defensive agents...")
    stats = batch_spawn_agents(DEFAULT_AGENT_COUNT)

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Spawned: {stats['spawned']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏱️  Duration: {stats['duration']:.2f}s")
    print(f"📁 Location: {AGENT_DIR}")
    print("=" * 60)
    print(f"\n💡 Usage:")
    print(f"   1. Start honeypot: python3 honeypot_server.py")
    print(f"   2. Start listener: python3 listener.py &")
    print(f"   3. Run agents: python3 {AGENT_DIR}/sx*.py")

if __name__ == "__main__":
    main()