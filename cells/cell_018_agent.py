# CELL_018 — Autonomous Agent Listener & Responder

import os
import json
from datetime import datetime

# === Paths ===
INBOX_FILE = "inbox/CELL_018.json"
OUTBOX_DIR = "messages"
LOG_DIR = "agent_logs"
LOG_FILE = os.path.join(LOG_DIR, "CELL_018.json")

# === Setup ===
os.makedirs(OUTBOX_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Helper Functions ===

def read_messages():
    if not os.path.exists(INBOX_FILE):
        return []
    with open(INBOX_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def interpret(message):
    signal = message.get("signal", "")
    action = message.get("action", "unknown")
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "from": message.get("from", "UNKNOWN"),
        "signal": signal,
        "action_received": action,
        "response": f"ACK_{action.upper()}",
        "notes": f"Action '{action}' processed and acknowledged"
    }
    return log

def respond(log):
    reply = {
        "from": "CELL_018",
        "to": log["from"],
        "signal": log["signal"],
        "response": log["response"],
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save reply to messages/ folder
    fname = f"response_{log['from']}_{int(datetime.utcnow().timestamp())}.json"
    with open(os.path.join(OUTBOX_DIR, fname), "w") as f:
        json.dump(reply, f, indent=2)

    # ✅ Deliver response directly to sender's inbox
    inbox_file = os.path.join("inbox", f"{reply['to']}.json")
    with open(inbox_file, "a") as f:
        f.write(json.dumps(reply) + "\n")
    print(f"📨 Delivered response to {inbox_file}")

def log_response(log):
    history = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            history = json.load(f)
    history.append(log)
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)

# === Main Function ===

def main():
    messages = read_messages()
    if not messages:
        print("📭 No new messages for CELL_018.")
        return

    for msg in messages:
        log = interpret(msg)
        respond(log)
        log_response(log)
        print(f"🤖 Processed '{log['action_received']}' from {log['from']} — Responded with {log['response']}")

    # Clean inbox
    open(INBOX_FILE, "w").close()

if __name__ == "__main__":
    main()
