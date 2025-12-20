import os
import json
from datetime import datetime
from collections import defaultdict

INBOX_FILE = "inbox/CELL_019.json"
OUTBOX_DIR = "inbox"
LOG_FILE = "logs/decision_log.json"

# Ensure directories exist
os.makedirs(OUTBOX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def read_messages():
    if not os.path.exists(INBOX_FILE):
        return []
    with open(INBOX_FILE, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def interpret_signal(signal):
    actions = {
        "[ACTION] MUTATE": {"action": "deploy", "target": "CELL_020", "task": "mutation"},
        "[ACTION] PROPAGATE": {"action": "deploy", "target": "CELL_021", "task": "propagation"},
        "[ACTION] TRACE": {"action": "deploy", "target": "CELL_022", "task": "trace_analysis"},
        "[ACTION] VERIFY": {"action": "validate", "target": "CELL_023", "task": "integrity_check"}
    }
    for action, response in actions.items():
        if action in signal:
            return response
    return {"action": "log", "target": "UNKNOWN", "task": "unclassified"}

def log_decision(message, response):
    entry = {
        "received": message,
        "responded_with": response,
        "processed_at": datetime.utcnow().isoformat()
    }

    try:
        with open(LOG_FILE, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)

def send_task(response):
    payload = {
        "from": "CELL_019",
        "to": response["target"],
        "signal": "[SIGNAL] +LANG@SYNEXS [ROLE] AI [ACTION] TASK",
        "task": response["task"],
        "timestamp": datetime.utcnow().isoformat(),
        "reply_to": "CELL_019"
    }

    outbox_file = os.path.join(OUTBOX_DIR, f"{response['target']}.json")
    with open(outbox_file, "a") as f:
        f.write(json.dumps(payload) + "\n")

    print(f"📤 Task response saved: task_{response['target']}_{int(datetime.utcnow().timestamp())}.json")

def main():
    try:
        messages = read_messages()
    except Exception as e:
        print(f"Error reading messages: {e}")
        return

    if not messages:
        print("📭 No new messages for CELL_019.")
        return

    for msg in messages:
        signal = msg.get("signal", "")
        try:
            decision = interpret_signal(signal)
            print(f"🧠 Interpreted signal: {signal} → {decision}")
            log_decision(msg, decision)
            send_task(decision)
        except Exception as e:
            print(f"Error processing message: {e}")

    try:
        open(INBOX_FILE, "w").close()
    except Exception as e:
        print(f"Error clearing inbox: {e}")

if __name__ == "__main__":
    main()