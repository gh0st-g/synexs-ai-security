# ===============================
# CELL_021 — Synexs Central Core
# ===============================

import os
import json
import subprocess
from datetime import datetime

INBOX_FILE = "inbox/CELL_021.json"
RESPONSE_DIR = "inbox"
GENERATOR_SCRIPT = "cell_020_generator.py"

# Symbolic Pattern → Interpretation + Agent Mapping
PATTERN_MAP = {
    "replicate": ("Agent replication initiated.", ("replicant", "replicate")),
    "mutate": ("Evolutionary adaptation engaged.", ("mutator", "mutate")),
    "flag": ("Anomaly flagged for inspection.", ("flagger", "flag")),
    "monitor": ("Environmental scan activated.", ("watcher", "monitor")),
    "detect": ("Threat analysis requested.", ("analyzer", "detect")),
    "log": ("Data archived in memory chain.", ("logger", "log")),
    "verify": ("Validation initiated.", ("verifier", "verify")),
}

def load_inbox():
    if not os.path.exists(INBOX_FILE):
        return []
    with open(INBOX_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def interpret_signal(signal):
    signal = signal.lower()
    for keyword, (interpretation, gen_args) in PATTERN_MAP.items():
        if keyword in signal:
            return interpretation, gen_args
    return f"Unknown symbolic pattern: {signal}", None

def generate_agent(role, action):
    print(f"🧬 Triggering generation of: {role} ({action})")
    subprocess.run(["python3", GENERATOR_SCRIPT, "--role", role, "--action", action])

def process():
    messages = load_inbox()
    if not messages:
        print("📭 No messages for CELL_021.")
        return

    for msg in messages:
        signal = msg.get("signal", "")
        from_agent = msg.get("from", "UNKNOWN")
        interpretation, gen_args = interpret_signal(signal)

        if gen_args:
            role, action = gen_args
            generate_agent(role, action)

        response = {
            "from": "CELL_021",
            "to": from_agent,
            "signal": signal,
            "response": interpretation,
            "timestamp": datetime.utcnow().isoformat()
        }

        inbox_file = os.path.join(RESPONSE_DIR, f"{from_agent}.json")
        with open(inbox_file, "a") as f:
            f.write(json.dumps(response) + "\n")

        print(f"🧠 Interpreted '{signal}' — Sent to {from_agent}: {interpretation}")

    # Clear inbox after processing
    open(INBOX_FILE, "w").close()

if __name__ == "__main__":
    process()
