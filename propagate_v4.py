import os
import json
import random
import time
import urllib.parse
from typing import Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== Configuration ====================
WORK_DIR = Path("/root/synexs")
ATTACK_PROFILES = WORK_DIR / "attack_profiles.json"
AGENT_DIR = WORK_DIR / "datasets" / "agents"
LOG_DIR = WORK_DIR / "datasets" / "logs"

HONEYPOT_URL = "http://127.0.0.1:8080"
DEFAULT_AGENT_COUNT = 50
MAX_WORKERS = 10

# Ensure directories exist
AGENT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_LOG_FILE = LOG_DIR / "attacks_log.jsonl"
RESULT_LOG_FILE = LOG_DIR / "agent_results.jsonl"

# Common realistic User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

# ==================== Configuration Loading ====================
def load_attack_profiles() -> Dict:
    try:
        with open(ATTACK_PROFILES, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  attack_profiles.json not found – create one with attack_categories")
        return {"attack_categories": {}, "evasion_techniques": {}}
    except Exception as e:
        print(f"⚠️  Failed to load profiles: {e}")
        return {"attack_categories": {}, "evasion_techniques": {}}

# ==================== Evasion Helpers ====================
def apply_encoding(payload: str, encoding: str) -> str:
    encoders = {
        "none": lambda p: p,
        "url": urllib.parse.quote,
        "double_url": lambda p: urllib.parse.quote(urllib.parse.quote(p)),
        "unicode": lambda p: ''.join(f'\\u{ord(c):04x}' for c in p),
        "html_entity": lambda p: ''.join(f'&#{ord(c)};' for c in p),
    }
    return encoders.get(encoding, lambda p: p)(payload)

def apply_case_variation(payload: str, mode: str) -> str:
    variations = {
        "none": lambda p: p,
        "upper": str.upper,
        "lower": str.lower,
        "alternating": lambda p: ''.join(c.upper() if i % 2 else c.lower() for i, c in enumerate(p)),
        "mixed": lambda p: ''.join(random.choice([str.upper, str.lower])(c) for c in p),
    }
    return variations.get(mode, lambda p: p)(payload)

# ==================== Attack Generators ====================
ATTACK_GENERATORS = {
    "sql_injection": lambda config, evasion: generate_sql_injection(config, evasion),
    "xss_injection": lambda config, evasion: generate_xss_injection(config, evasion),
    "path_traversal": lambda config, evasion: generate_path_traversal(config, evasion),
    "command_injection": lambda config, evasion: generate_command_injection(config, evasion),
    "api_abuse": lambda config, evasion: generate_api_abuse(config, evasion),
    "authentication_bypass": lambda config, evasion: generate_authentication_bypass(config, evasion),
    "directory_scanning": lambda config, evasion: generate_directory_scanning(config, evasion),
    "rate_limit_test": lambda config, evasion: generate_rate_limit_test(config, evasion),
    "crawler_impersonation": lambda config, evasion: generate_crawler_impersonation(config, evasion),
    "http_method_abuse": lambda config, evasion: generate_http_method_abuse(config, evasion),
    "legitimate_traffic": lambda config, evasion: generate_legitimate_traffic(config, evasion),
}

# ==================== Logging ====================
def log_attack(attack: Dict, agent_id: str):
    entry = {
        "agent_id": agent_id,
        "timestamp": time.time(),
        "attack_type": attack["type"],
        "endpoint": attack["endpoint"],
        "method": attack["method"],
        "raw_payload": attack.get("raw_payload"),
        "description": attack.get("description")
    }
    try:
        with open(ATTACK_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"❌ Error logging attack: {e}")

def log_result(status, url, elapsed, agent_id):
    entry = {"agent_id": agent_id, "status": status, "url": url, "elapsed": elapsed, "timestamp": time.time()}
    try:
        with open(RESULT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"❌ Error logging result: {e}")

# ==================== Agent Generation ====================
def generate_agent_code(agent_id: str, attack: Dict) -> str:
    ua_line = f'"{random.choice(USER_AGENTS)}"'
    if attack["type"] == "crawler_impersonation":
        bot = attack.get("fake_bot", "googlebot").lower()
        bot_agents = {
            "googlebot": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "bingbot": "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
            "yahoobot": "Mozilla/5.0 (compatible; Yahoo! Slurp; http://help.yahoo.com/help/us/ysearch/slurp)",
        }
        ua_line = f'"{bot_agents.get(bot, bot_agents["googlebot"])}"'

    post_data = attack.get("post_data")
    post_code = f'\nDATA = {json.dumps(post_data)}' if post_data else ""
    post_json = ", json=DATA" if post_data else ""

    paths_code = '["/"]'
    if attack["type"] in ["legitimate_traffic", "crawler_impersonation"]:
        common_paths = ["/", "/about", "/contact", "/products", "/blog", "/search?q=test", "/robots.txt", "/sitemap.xml", "/login", "/api/data"]
        steps = attack.get("session_steps", 6)
        paths_code = json.dumps(random.sample(common_paths, min(steps, len(common_paths))))

    execution_block = "execute_single(URL)"
    if attack["type"] == "rate_limit_test":
        burst = attack.get("burst_size", 30)
        execution_block = f"""
    print(f"[{agent_id}] Starting burst of {burst} requests...")
    for _ in range({burst}):
        execute_single(URL)
        time.sleep(random.uniform(0.01, 0.15))
"""
    elif attack["type"] in ["legitimate_traffic", "crawler_impersonation"]:
        delay_min, delay_max = (1.0, 7.0) if attack["type"] == "legitimate_traffic" else (0.5, 3.0)
        execution_block = f"""
    paths = {paths_code}
    for path in paths:
        full_url = HONEYPOT_URL + path
        execute_single(full_url)
        time.sleep(random.uniform({delay_min}, {delay_max}))
"""

    template = f'''# Purple Team Agent: {agent_id}
# Type: {attack["type"]}
# Description: {attack.get("description", "")}

import requests
import time
import random
import json

URL = "{HONEYPOT_URL}{attack["endpoint"]}"
METHOD = "{attack["method"]}"{post_code}

USER_AGENTS = {USER_AGENTS}

HEADERS = {{
    "User-Agent": {ua_line},
    "Accept": "*/*",
    "Referer": random.choice(["https://www.google.com/", "https://www.bing.com/", "https://search.yahoo.com/"]),
    "Connection": "close"
}}

def execute_single(url):
    start = time.time()
    try:
        if METHOD == "GET":
            r = requests.get(url, headers=HEADERS, timeout=10)
        elif METHOD == "POST":
            h = HEADERS.copy()
            h["Content-Type"] = "application/json"
            r = requests.post(url, headers=h{post_json}, timeout=10)
        else:
            r = requests.request(METHOD, url, headers=HEADERS, timeout=10)
        elapsed = time.time() - start
        print(f"[{agent_id}] {{r.status_code}} {{METHOD}} {{url[:80]}}")
        log_result(r.status_code, url, elapsed, "{agent_id}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{agent_id}] ERROR {{e}}")
        log_result("ERROR", url, elapsed, "{agent_id}")

if __name__ == "__main__":
    time.sleep(random.uniform(0.5, 6.0))
{execution_block}
'''
    return template

def spawn_agent(agent_id: str, attack: Dict) -> Tuple[str, bool, Dict]:
    try:
        code = generate_agent_code(agent_id, attack)
        path = AGENT_DIR / f"{agent_id}.py"
        with open(path, "w") as f:
            f.write(code)
        log_attack(attack, agent_id)
        return (agent_id, True, attack)
    except Exception as e:
        print(f"❌ {agent_id}: Spawn failed – {e}")
        return (agent_id, False, attack)

# ==================== Batch Spawner ====================
def batch_spawn_agents(count: int = DEFAULT_AGENT_COUNT) -> Dict:
    start_time = time.time()
    profiles = load_attack_profiles()
    evasion = profiles.get("evasion_techniques", {})

    attack_patterns = []
    attack_stats = {}

    for _ in range(count):
        attack_type, config = random.choice(list(profiles["attack_categories"].items()))
        gen_func = ATTACK_GENERATORS.get(attack_type)
        if gen_func:
            attack = gen_func(config, evasion)
            attack_patterns.append(attack)
            attack_stats[attack_type] = attack_stats.get(attack_type, 0) + 1
        else:
            print(f"⚠️  Unknown attack type: {attack_type}")

    agent_tasks = [
        (f"agent_{int(time.time())}_{i:05d}", attack)
        for i, attack in enumerate(attack_patterns)
    ]

    spawned = failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(spawn_agent, aid, atk): aid for aid, atk in agent_tasks}
        for future in as_completed(futures):
            _, success, _ = future.result()
            spawned += success
            failed += not success

    duration = time.time() - start_time
    print(f"✅ Spawning complete: {spawned} agents created, {failed} failed ({duration:.2f}s)")
    return {"spawned": spawned, "failed": failed, "duration": duration, "distribution": attack_stats}

# ==================== Main ====================
def main():
    print("🚀 Synexs Purple Team Traffic Generator – Starting")
    results = batch_spawn_agents(DEFAULT_AGENT_COUNT)
    print("\n📊 Summary:")
    print(json.dumps(results, indent=2))
    print(f"\n📜 Attack metadata: {ATTACK_LOG_FILE}")
    print(f"📊 Response results: {RESULT_LOG_FILE}")

if __name__ == "__main__":
    main()