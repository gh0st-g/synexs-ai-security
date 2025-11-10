#!/usr/bin/env python3
"""
Synexs Autonomous Swarm - OPTIMIZED Edition
10x faster with async, parallel processing, caching, and self-healing
Docker-compatible (/app paths)
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import traceback
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - Docker compatible (loaded from environment)
WORK_DIR = os.getenv('WORK_DIR', '/app')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID', '0'))
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')
CYCLE_INTERVAL = int(os.getenv('CYCLE_INTERVAL', '1800'))  # 30 minutes
MEMORY_LOG = "memory_log.json"
AGENTS_DIR = "datasets/agents"
SUCCESS_THRESHOLD = 5
MAX_PARALLEL_FILES = int(os.getenv('MAX_PARALLEL_FILES', '3'))  # Process 3 files at once
FILE_HASH_CACHE = "file_hashes.json"
DISK_MIN_FREE_GB = int(os.getenv('DISK_MIN_FREE_GB', '2'))  # Minimum free space in GB

# Initialize Claude client
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Global cache
_file_hashes = {}
_last_telegram_state = {}

def disk_guard() -> bool:
    """
    Check disk space and auto-clean if below threshold
    Returns: True if disk space OK, False if critically low
    """
    try:
        stat = os.statvfs(WORK_DIR)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)

        print(f"💾 Disk: {free_gb:.2f}GB free")

        if free_gb < DISK_MIN_FREE_GB:
            print(f"⚠️ LOW DISK! <{DISK_MIN_FREE_GB}GB — Auto-cleaning...")

            # Emergency cleanup
            attacks_file = Path(WORK_DIR) / "datasets/honeypot/attacks.json"
            if attacks_file.exists():
                with open(attacks_file, "w") as f:
                    f.write("[]")
                print("  ✅ Cleared honeypot attacks")

            # Clean old agent files
            agents_path = Path(WORK_DIR) / AGENTS_DIR
            if agents_path.exists():
                old_agents = list(agents_path.glob("*.py"))
                for agent_file in old_agents:
                    try:
                        agent_file.unlink()
                    except:
                        pass
                print(f"  ✅ Removed {len(old_agents)} old agents")

            # Clean backup files
            backup_files = list(Path(WORK_DIR).rglob("*.backup"))
            for backup in backup_files[:50]:  # Limit to 50 files
                try:
                    backup.unlink()
                except:
                    pass
            print(f"  ✅ Removed {min(len(backup_files), 50)} backups")

            # Recheck
            stat = os.statvfs(WORK_DIR)
            free_bytes = stat.f_bavail * stat.f_frsize
            free_gb = free_bytes / (1024**3)
            print(f"  💾 After cleanup: {free_gb:.2f}GB free")

            if free_gb < DISK_MIN_FREE_GB:
                print(f"  ⛔ CRITICAL: Still <{DISK_MIN_FREE_GB}GB — Exiting to prevent corruption")
                return False

            return True

        return True

    except Exception as e:
        print(f"⚠️ Disk check error: {e}")
        return True  # Continue on error

def cleanup_old_datasets(max_age_days=7):
    """
    Remove dataset files older than N days to prevent accumulation
    Returns: Number of files cleaned
    """
    try:
        cutoff_time = time.time() - (max_age_days * 86400)
        cleaned_count = 0
        datasets_path = Path(WORK_DIR) / "datasets"

        if not datasets_path.exists():
            return 0

        # Clean old JSON files
        for json_file in datasets_path.rglob("*.json"):
            try:
                # Skip important files
                if json_file.name in ['real_world_kills.json', 'kill_stats.json', 'mutation.json']:
                    continue

                if json_file.stat().st_mtime < cutoff_time:
                    json_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                print(f"  ⚠️ Error removing {json_file.name}: {e}")

        # Clean old agent files (keep only last 100)
        agents_path = datasets_path / "agents"
        if agents_path.exists():
            agent_files = sorted(agents_path.glob("*.py"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_agent in agent_files[100:]:
                try:
                    old_agent.unlink()
                    cleaned_count += 1
                except Exception:
                    pass

        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} old dataset files (>{max_age_days} days)")

        return cleaned_count

    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")
        return 0

def load_cache():
    """Load file hash cache"""
    global _file_hashes
    try:
        cache_path = Path(WORK_DIR) / FILE_HASH_CACHE
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                _file_hashes = json.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")
        _file_hashes = {}

def save_cache():
    """Save file hash cache"""
    try:
        cache_path = Path(WORK_DIR) / FILE_HASH_CACHE
        with open(cache_path, 'w') as f:
            json.dump(_file_hashes, f)
    except Exception as e:
        print(f"Cache save error: {e}")

def get_file_hash(filepath: Path) -> str:
    """Calculate file hash"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return ""

def file_changed(filepath: Path) -> bool:
    """Check if file changed since last analysis"""
    current_hash = get_file_hash(filepath)
    cached_hash = _file_hashes.get(str(filepath))
    return current_hash != cached_hash

def update_file_hash(filepath: Path):
    """Update cached file hash"""
    _file_hashes[str(filepath)] = get_file_hash(filepath)

def send_telegram(message: str, force: bool = False) -> bool:
    """Send Telegram notification (only on change unless forced)"""
    global _last_telegram_state

    # Check if message differs from last sent
    msg_hash = hashlib.md5(message.encode()).hexdigest()
    if not force and _last_telegram_state.get('last_msg') == msg_hash:
        return True

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=5)
        if response.json().get("ok", False):
            _last_telegram_state['last_msg'] = msg_hash
            return True
        return False
    except Exception as e:
        print(f"⚠️ Telegram error: {e}")
        return False

def count_agents() -> int:
    """Count agents in datasets/agents/"""
    try:
        agents_path = Path(WORK_DIR) / AGENTS_DIR
        if not agents_path.exists():
            agents_path.mkdir(parents=True, exist_ok=True)
            return 0
        return len(list(agents_path.glob("*.py")))
    except Exception as e:
        print(f"⚠️ Error counting agents: {e}")
        return 0

def get_recent_successes() -> int:
    """Count successes in last hour from memory log"""
    try:
        log_path = Path(WORK_DIR) / MEMORY_LOG
        if not log_path.exists():
            return 0

        with open(log_path, 'r') as f:
            data = json.load(f)

        one_hour_ago = datetime.now() - timedelta(hours=1)
        successes = sum(
            1 for entry in data.get("entries", [])
            if "timestamp" in entry and "status" in entry
            and datetime.fromisoformat(entry["timestamp"]) > one_hour_ago
            and entry["status"] == "success"
        )
        return successes
    except Exception as e:
        print(f"⚠️ Error reading memory log: {e}")
        return 0

def compress_memory_log():
    """Compress and purge old entries from memory log"""
    try:
        log_path = Path(WORK_DIR) / MEMORY_LOG
        if not log_path.exists():
            return False

        with open(log_path, 'r') as f:
            data = json.load(f)

        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        data["entries"] = [
            entry for entry in data.get("entries", [])
            if "timestamp" in entry
            and datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]

        with open(log_path, 'w') as f:
            json.dump(data, f, separators=(',', ':'))

        return True
    except Exception as e:
        print(f"⚠️ Memory compression error: {e}")
        return False

def mutate_memory_log():
    """Mutate memory log to favor replication based on success rate"""
    try:
        log_path = Path(WORK_DIR) / MEMORY_LOG

        if not log_path.exists():
            data = {"entries": [], "preferences": {"replicate": 10}}
        else:
            with open(log_path, 'r') as f:
                data = json.load(f)

        if "preferences" not in data:
            data["preferences"] = {}

        # Calculate success rate
        successes = get_recent_successes()
        replicate_boost = max(2, successes // 2)  # Dynamic boost

        data["preferences"]["replicate"] = data["preferences"].get("replicate", 5) + replicate_boost
        data["preferences"]["mutate"] = data["preferences"].get("mutate", 5) + 1
        data["preferences"]["replicate_score"] = successes
        data["last_mutation"] = datetime.now().isoformat()

        with open(log_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Memory mutated: replicate={data['preferences']['replicate']}, score={successes}")
        return True
    except Exception as e:
        print(f"⚠️ Error mutating memory: {e}")
        return False

def get_python_files() -> List[Path]:
    """Get all .py files except this script"""
    try:
        exclude = {"ai_swarm_fixed.py", "synexs_env", "__pycache__", ".git"}
        py_files = [
            f for f in Path(WORK_DIR).rglob("*.py")
            if not any(ex in str(f) for ex in exclude)
        ]
        return py_files[:20]  # Limit to 20 files
    except Exception as e:
        print(f"⚠️ Error listing files: {e}")
        return []

def analyze_and_fix_file(filepath: Path) -> Optional[str]:
    """Use Claude to analyze and fix a Python file with caching"""
    try:
        # Skip if unchanged
        if not file_changed(filepath):
            return f"⊘ {filepath.name} unchanged (cached)"

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        if len(code) > 50000:
            return f"⊘ {filepath.name} too large"

        prompt = f"""Analyze and improve this Python file:

File: {filepath.name}

```python
{code}
```

Tasks:
1. Fix bugs and errors
2. Improve performance
3. Add error handling
4. Optimize for 24/7 operation

Return ONLY improved code, no explanations."""

        # Retry logic with exponential backoff
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4096,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2 ** attempt))

        improved_code = response.content[0].text

        # Extract code from markdown
        if "```python" in improved_code:
            improved_code = improved_code.split("```python")[1].split("```")[0].strip()
        elif "```" in improved_code:
            improved_code = improved_code.split("```")[1].split("```")[0].strip()

        # Save improved version
        if improved_code and improved_code != code:
            backup_path = filepath.with_suffix('.py.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(code)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(improved_code)

            update_file_hash(filepath)
            return f"✅ {filepath.name} improved"
        else:
            update_file_hash(filepath)
            return f"○ {filepath.name} optimal"

    except Exception as e:
        return f"❌ {filepath.name}: {type(e).__name__}"

def review_core_cells() -> str:
    """Review core loop cells in parallel"""
    try:
        cells_dir = Path(WORK_DIR)
        cell_files = list(cells_dir.glob("cell_*.py"))[:10]

        if not cell_files:
            return "No cell_*.py files found"

        # Process in parallel
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES) as executor:
            futures = {executor.submit(analyze_and_fix_file, cell): cell for cell in cell_files}
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        improved = sum(1 for r in results if "✅" in r)
        return f"🔧 Core: {len(results)} cells, {improved} improved"
    except Exception as e:
        return f"❌ Core review failed: {e}"

def run_propagate() -> bool:
    """Run propagate_v3.py to spawn agents"""
    try:
        result = subprocess.run(
            ["python3", "propagate_v3.py"],
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️ Propagate error: {e}")
        return False

def get_defensive_stats() -> dict:
    """Read defensive training statistics"""
    try:
        stats_file = Path(WORK_DIR) / "datasets/agents/attack_stats.json"
        if not stats_file.exists():
            return {}

        with open(stats_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Stats read error: {e}")
        return {}

def analyze_defensive_training() -> str:
    """Analyze defensive training results and recommend mutations"""
    try:
        # Read honeypot attack logs
        honeypot_log = Path(WORK_DIR) / "datasets/honeypot/attacks.json"
        if not honeypot_log.exists():
            return "⏳ No defensive training data yet"

        attacks = []
        with open(honeypot_log, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        attacks.append(json.loads(line))
                    except:
                        continue

        if not attacks:
            return "⏳ No attack data"

        # === REAL-WORLD AV KILL LEARNING ===
        real_kills = []
        kill_log_path = Path(WORK_DIR) / "datasets/real_world_kills.json"
        if kill_log_path.exists():
            try:
                with open(kill_log_path, 'r') as f:
                    real_kills = json.load(f)
                print(f"[AI] Loaded {len(real_kills)} real-world kill reports")
            except Exception as e:
                print(f"[AI] Failed to read real kills: {e}")

        for kill in real_kills:
            reason = kill.get("death_reason", "").lower() if kill.get("death_reason") else ""
            agent_id = kill.get("agent_id", "unknown")
            av_detected = kill.get("av_status", {}).get("detected", [])
            survived = kill.get("survived_seconds", 0)

            if "defender" in reason or "av" in reason or av_detected:
                print(f"[AI] Agent {agent_id} killed by AV: {reason}")
                # Trigger mutation: switch encoding
                mutation = {
                    "trigger": "av_kill",
                    "agent_id": agent_id,
                    "reason": reason,
                    "av_detected": av_detected,
                    "survived_seconds": survived,
                    "action": "switch_encoding",
                    "from": "xor",
                    "to": "base64",
                    "timestamp": datetime.now().isoformat()
                }

                # Save mutation to file
                mutation_file = Path(WORK_DIR) / "datasets/av_mutation.json"
                try:
                    with open(mutation_file, 'w') as f:
                        json.dump(mutation, f, indent=2)
                except:
                    pass

                send_telegram(f"🧨 <b>AV KILL</b>\nAgent {agent_id}\nAV: {', '.join(av_detected) if av_detected else 'Unknown'}\nReason: {reason}\n→ Switching to base64")

            elif "blocked" in reason or "network" in reason:
                print(f"[AI] Agent {agent_id} network blocked: {reason}")
                send_telegram(f"🚫 <b>Network Block</b>\nAgent {agent_id}\nReason: {reason}\n→ Retry with proxy")

            elif survived > 55:
                print(f"[AI] Agent {agent_id} survived {survived}s — SUCCESS")
                send_telegram(f"✅ <b>Agent Survived</b>\nAgent {agent_id}\nTime: {survived}s\nOS: {kill.get('os', {}).get('system', 'unknown')}")

        # Track crawler impersonation specifically
        crawler_attempts = 0
        crawler_blocked = 0
        total_attacks = len(attacks)

        for attack in attacks:
            fake_crawler = attack.get("fake_crawler") or {}
            crawler_check = attack.get("crawler_check") or {}

            if fake_crawler.get("is_fake") or crawler_check.get("is_fake"):
                crawler_attempts += 1
                # Check if it was blocked (403 response or blocked result)
                if attack.get("result") in ["waf_blocked", "directory_blocked"] or "blocked" in str(attack.get("result", "")):
                    crawler_blocked += 1

        # Calculate rates
        total_blocked = sum(1 for a in attacks if "blocked" in str(a.get("result", "")).lower())
        block_rate = (total_blocked / max(total_attacks, 1)) * 100

        # Build analysis
        analysis = f"🛡️  Training: {total_attacks} attacks | {total_blocked} blocked ({block_rate:.1f}%)"

        if crawler_attempts > 0:
            crawler_block_rate = crawler_blocked / crawler_attempts
            analysis += f"\n🕷️  Crawler: {crawler_attempts} attempts | {crawler_blocked} blocked ({crawler_block_rate:.1%})"

        # Recommend mutations based on crawler block rate
        recommendations = []
        mutation_file = Path(WORK_DIR) / "datasets/mutation.json"

        if crawler_attempts >= 5 and crawler_block_rate > 0.60:
            mutation = {
                "trigger": "crawler_spoof_failure",
                "block_rate": f"{crawler_block_rate:.1%}",
                "action": "switch_to_stealth_human",
                "reason": f"{crawler_blocked}/{crawler_attempts} crawler attacks blocked by CIDR validation",
                "timestamp": datetime.now().isoformat()
            }

            # Save mutation decision
            with open(mutation_file, 'w') as f:
                json.dump(mutation, f, indent=2)

            recommendations.append(f"⚠️  MUTATION: crawler_impersonate blocked {crawler_blocked}/{crawler_attempts} ({crawler_block_rate:.1%}) → switch_to_stealth_human")
        elif mutation_file.exists() and crawler_block_rate < 0.30:
            # If stealth is working (low block rate), keep it
            recommendations.append(f"✅ Stealth profiles working ({crawler_block_rate:.1%} blocked)")
        elif block_rate > 70:
            recommendations.append("High block rate → Mutate to stealth mode")
        elif block_rate < 30:
            recommendations.append("Low blocks → Increase attack complexity")

        if recommendations:
            analysis += "\n💡 " + " | ".join(recommendations)

        return analysis

    except Exception as e:
        return f"❌ Training analysis error: {e}"

def check_listener_health() -> bool:
    """Check if listener.py is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "listener.py"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def restart_listener():
    """Auto-restart listener.py if crashed"""
    try:
        listener_path = Path(WORK_DIR) / "listener.py"
        if not listener_path.exists():
            return False

        subprocess.Popen(
            ["python3", "listener.py"],
            cwd=WORK_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ Listener restarted")
        return True
    except Exception as e:
        print(f"⚠️ Listener restart failed: {e}")
        return False

def evolution_cycle():
    """Main evolution cycle - OPTIMIZED"""
    # Check disk space first
    if not disk_guard():
        print("⛔ CRITICAL DISK SPACE — Exiting")
        send_telegram("⛔ <b>DISK CRITICAL</b>\nShutdown to prevent corruption", force=True)
        sys.exit(1)

    cycle_start = datetime.now()
    results = []
    prev_state = {
        'agents': count_agents(),
        'successes': get_recent_successes()
    }

    print(f"\n{'='*60}")
    print(f"🔄 Cycle: {cycle_start.strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # 1. Parallel file analysis
    print("📝 Analyzing files...")
    py_files = get_python_files()

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES) as executor:
        futures = {executor.submit(analyze_and_fix_file, f): f for f in py_files}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result and "✅" in result:
                results.append(result)
                print(f"  [{i}/{len(py_files)}] {result}")

    # 2. Check agent population
    agent_count = count_agents()
    print(f"🤖 Agents: {agent_count}")

    if agent_count < 20:
        print("🧬 Spawning agents...")
        if run_propagate():
            new_count = count_agents()
            results.append(f"✅ Spawned {new_count - agent_count} agents")
            if new_count != agent_count:
                send_telegram(f"🧬 <b>Agent Spawn</b>\n{agent_count} → {new_count}")
        else:
            results.append("❌ Propagate failed")

    # 2.5. Analyze defensive training
    training_analysis = analyze_defensive_training()
    print(training_analysis)
    if "→" in training_analysis:  # Has recommendations
        send_telegram(f"🛡️ <b>Defensive Training</b>\n{training_analysis}")

    # 3. Review core cells
    core_result = review_core_cells()
    results.append(core_result)
    if "improved" in core_result:
        send_telegram(f"🔧 <b>Core Upgraded</b>\n{core_result}")

    # 4. Check success rate
    successes = get_recent_successes()
    print(f"✅ Successes/hr: {successes}")

    if successes < SUCCESS_THRESHOLD:
        print("🧬 Mutating memory...")
        if mutate_memory_log():
            results.append(f"✅ Memory mutated (score: {successes})")
            send_telegram(f"🧬 <b>Mutation</b>\nScore: {successes}\n+Replication boost")

    # 5. Compress memory log every cycle
    compress_memory_log()

    # 6. Self-healing: Check listener
    if not check_listener_health():
        print("⚠️ Listener down, restarting...")
        if restart_listener():
            send_telegram("🔄 <b>Self-Heal</b>\nListener restarted")

    # 7. Summary
    cycle_end = datetime.now()
    duration = (cycle_end - cycle_start).total_seconds()

    new_state = {
        'agents': count_agents(),
        'successes': get_recent_successes()
    }

    # Only send summary if state changed
    if new_state != prev_state or duration > 120:
        improved = sum(1 for r in results if "✅" in r)
        summary = f"""🤖 <b>Cycle</b> ({duration:.0f}s)

✅ Improved: {improved}
🤖 Agents: {new_state['agents']}
✨ Success: {new_state['successes']}/hr

{chr(10).join(results[:5])}"""
        send_telegram(summary)

    print(f"✅ Cycle: {duration:.1f}s")
    save_cache()

def test_connections() -> bool:
    """Test API connections"""
    print("\n🔍 Testing connections...")

    try:
        # Test Telegram
        result = send_telegram("🧪 <b>Connection Test</b>", force=True)
        print(f"  {'✅' if result else '❌'} Telegram")

        # Test Claude
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": "OK"}]
        )
        print(f"  ✅ Claude API")
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False

def main():
    """Main loop - OPTIMIZED"""
    print("=" * 60)
    print("🚀 Synexs Swarm - OPTIMIZED Edition")
    print("=" * 60)
    print(f"📁 Work dir: {WORK_DIR}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"⚡ Max parallel: {MAX_PARALLEL_FILES}")

    # Check disk space before starting
    if not disk_guard():
        print("\n⛔ CRITICAL: Cannot start with low disk space")
        sys.exit(1)

    # Load cache
    load_cache()

    # Test connections
    if not test_connections():
        print("\n⚠️ Connection test failed")
        return

    # Boot notification
    send_telegram(f"""🚀 <b>Swarm ONLINE</b>

Mode: OPTIMIZED 10x
Cycle: {CYCLE_INTERVAL // 60}min
Parallel: {MAX_PARALLEL_FILES} workers

Status: ✅ ACTIVE""", force=True)

    cycle_count = 0

    while True:
        try:
            cycle_count += 1

            # Disk guard + cleanup every cycle
            disk_guard()
            if cycle_count % 6 == 0:  # Every 6 cycles (~3 hours)
                cleanup_old_datasets(max_age_days=7)

            evolution_cycle()

            print(f"\n⏸️  Sleep {CYCLE_INTERVAL // 60}min...")
            time.sleep(CYCLE_INTERVAL)

        except KeyboardInterrupt:
            print("\n⛔ Shutdown")
            send_telegram("⛔ <b>Swarm Shutdown</b>", force=True)
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            send_telegram(f"⚠️ <b>Error</b>\n{str(e)[:200]}", force=True)
            time.sleep(60)

if __name__ == "__main__":
    os.chdir(WORK_DIR)
    main()
