# cell_019_planner.py — Synexs Task Planner Agent
import os
import json
from datetime import datetime

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
    if "[ACTION] MUTATE" in signal:
        return {"action": "deploy", "target": "CELL_020", "task": "mutation"}
    elif "[ACTION] PROPAGATE" in signal:
        return {"action": "deploy", "target": "CELL_021", "task": "propagation"}
    elif "[ACTION] TRACE" in signal:
        return {"action": "deploy", "target": "CELL_022", "task": "trace_analysis"}
    elif "[ACTION] VERIFY" in signal:
        return {"action": "validate", "target": "CELL_023", "task": "integrity_check"}
    else:
        return {"action": "log", "target": "UNKNOWN", "task": "unclassified"}

def log_decision(message, response):
    entry = {
        "received": message,
        "responded_with": response,
        "processed_at": datetime.utcnow().isoformat()
    }

    history = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    history = data
        except json.JSONDecodeError:
            pass  # corrupted or invalid, reset safely

    history.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)

def send_task(response):
    target = response["target"]
    payload = {
        "from": "CELL_019",
        "to": target,
        "signal": "[SIGNAL] +LANG@SYNEXS [ROLE] AI [ACTION] TASK",
        "task": response["task"],
        "timestamp": datetime.utcnow().isoformat(),
        "reply_to": "CELL_019"
    }

    outbox_file = os.path.join(OUTBOX_DIR, f"{target}.json")
    with open(outbox_file, "a") as f:
        f.write(json.dumps(payload) + "\n")

    print(f"📤 Task response saved: task_{target}_{int(datetime.utcnow().timestamp())}.json")

def main():
    messages = read_messages()
    if not messages:
        print("📭 No new messages for CELL_019.")
        return

    for msg in messages:
        signal = msg.get("signal", "")
        decision = interpret_signal(signal)
        print(f"🧠 Interpreted signal: {signal} → {decision}")

        log_decision(msg, decision)
        send_task(decision)

    # Clear inbox after processing
    open(INBOX_FILE, "w").close()

if __name__ == "__main__":
    main()
