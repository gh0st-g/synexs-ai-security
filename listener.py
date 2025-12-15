# ~/synexs/listener.py
from utils.telegram_notify import send_telegram
import logging
import logging.handlers
import time
import signal
import sys
import os
import fcntl
import json
import threading
from datetime import datetime
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- External Intel ---
try:
    from utils.external_intel import shodan_lookup, nmap_scan
    from binary_protocol import encode_base64
    INTEL_AVAILABLE = True
except Exception as e:
    logging.warning(f"Intel module not available: {e}. Running in basic mode.")
    INTEL_AVAILABLE = False

# --- Database ---
try:
    from db.database import get_db
    from db.models import Attack
    DATABASE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Database module not available: {e}. Falling back to JSONL.")
    DATABASE_AVAILABLE = False

# PID & Logging
PID_FILE = '/tmp/listener.pid'
lock_fp = None
handler = logging.handlers.RotatingFileHandler('listener.log', maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logging.basicConfig(handlers=[handler], level=logging.INFO)

# Stats
running = True
message_count = error_count = intel_hits = training_samples_added = 0
start_time = last_message_time = datetime.now()
processed_ips = set()
intel_cache_dir = "/tmp/synexs_intel_cache"
os.makedirs(intel_cache_dir, exist_ok=True)

# Health Endpoint
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            uptime = (datetime.now() - start_time).total_seconds()
            data = {
                'status': 'healthy' if running else 'shutting_down',
                'uptime_seconds': uptime,
                'messages_processed': message_count,
                'errors': error_count,
                'last_message': last_message_time.isoformat() if last_message_time else None,
                'pid': os.getpid(),
                'intel_hits': intel_hits,
                'training_samples_added': training_samples_added,
                'processed_ips': len(processed_ips)
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args): pass

def start_health_server():
    try:
        HTTPServer(('0.0.0.0', 8765), HealthHandler).serve_forever()
    except Exception as e:
        logging.warning(f"Health server failed: {e}")

# PID Lock
def acquire_pid_lock():
    global lock_fp
    try:
        lock_fp = open(PID_FILE, 'x')
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fp.write(str(os.getpid()))
        lock_fp.flush()
        return True
    except IOError:
        logging.error("Another instance is running!")
        return False

def release_pid_lock():
    global lock_fp
    if lock_fp:
        try:
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
            lock_fp.close()
            os.remove(PID_FILE)
        except: pass

# Signal Handling
def signal_handler(sig, frame):
    global running
    logging.info("Shutting down...")
    running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- MAIN ENRICHMENT & ALERT ---
def enrich_and_train(ip: str):
    global intel_hits, training_samples_added
    if not INTEL_AVAILABLE or ip in processed_ips:
        return

    cache_file = os.path.join(intel_cache_dir, f"{ip}.json")
    intel = None

    # Load from cache or fetch
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                intel = json.load(f)
        except: pass

    if intel is None:
        try:
            logging.info(f"Enriching IP: {ip}")
            intel = {"ip": ip, "timestamp": int(datetime.now().timestamp())}
            intel.update(shodan_lookup(ip))
            intel.update({"nmap": nmap_scan(ip, quick=True)})
            with open(cache_file, "w") as f:
                json.dump(intel, f)
        except Exception as e:
            logging.error(f"Enrichment failed for {ip}: {e}")
            return

    # Generate training sample if CVE found
    vulns = intel.get("vulns", []) or []
    if vulns:
        actions = ["SCAN", "ATTACK", "REFINE"]
        binary = encode_base64(actions)

        # Prepare data
        instruction = f"Host {ip} has vuln {vulns[0]}"
        output = f"Execute: {' → '.join(actions)}"

        # Save to PostgreSQL (preferred) or JSONL (fallback)
        if DATABASE_AVAILABLE:
            try:
                with get_db() as db:
                    attack = Attack(
                        ip=ip,
                        timestamp=datetime.fromtimestamp(intel["timestamp"]),
                        vulns=vulns,
                        open_ports=intel.get("open_ports", []),
                        country=intel.get("country"),
                        country_code=intel.get("country_code"),
                        city=intel.get("city"),
                        latitude=intel.get("latitude"),
                        longitude=intel.get("longitude"),
                        actions=actions,
                        binary_input=binary,
                        protocol="v3",
                        format="base64",
                        instruction=instruction,
                        output=output,
                        source="shodan_nmap",
                        org=intel.get("org"),
                        isp=intel.get("isp"),
                        asn=intel.get("asn"),
                        is_threat=True,
                        severity="high" if len(vulns) > 2 else "medium"
                    )
                    db.add(attack)
                    db.commit()
                logging.info(f"Attack saved to DB: {ip} → {vulns[0]}")
            except Exception as e:
                logging.error(f"Database write failed: {e}, falling back to JSONL")
                # Fallback to JSONL
                sample = {
                    "instruction": instruction,
                    "input": f"binary:{binary}",
                    "output": output,
                    "actions": actions,
                    "protocol": "v3",
                    "format": "base64",
                    "source": "shodan_nmap",
                    "timestamp": intel["timestamp"]
                }
                with open("training_binary_v3.jsonl", "a") as f:
                    f.write(json.dumps(sample) + "\n")
        else:
            # JSONL fallback
            sample = {
                "instruction": instruction,
                "input": f"binary:{binary}",
                "output": output,
                "actions": actions,
                "protocol": "v3",
                "format": "base64",
                "source": "shodan_nmap",
                "timestamp": intel["timestamp"]
            }
            with open("training_binary_v3.jsonl", "a") as f:
                f.write(json.dumps(sample) + "\n")

        training_samples_added += 1
        logging.info(f"Training sample added: {ip} → {vulns[0]}")

    # --- TELEGRAM ALERT ---
    notify_details = {"vulns": vulns, "open_ports": intel.get("open_ports", [])}
    send_telegram(ip, notify_details)

    intel_hits += 1
    processed_ips.add(ip)

# --- Redis Listener ---
def listen():
    global message_count, error_count, last_message_time
    import redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()

    while running:
        try:
            msg = r.blpop('agent_tasks', timeout=5)
            if msg:
                data = json.loads(msg[1])
                src_ip = data.get("source_ip") or data.get("ip")
                if src_ip:
                    threading.Thread(target=enrich_and_train, args=(src_ip,), daemon=True).start()
                logging.info(f"Processed: {msg[1]}")
                message_count += 1
                last_message_time = datetime.now()
        except Exception as e:
            error_count += 1
            logging.error(f"Error: {e}")

# --- Main ---
def main():
    if not acquire_pid_lock(): return
    threading.Thread(target=start_health_server, daemon=True).start()
    try:
        logging.info("Synexs Listener STARTED")
        listen()
    finally:
        release_pid_lock()
        logging.info("Listener stopped")

if __name__ == '__main__':
    main()
