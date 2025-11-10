# cell_020_executor.py

import os
import json
import time
from datetime import datetime

INBOX_FILE = "inbox/CELL_020.json"
LOG_FILE = "logs/task_execution_log.json"
REPLY_ENABLED = True

def read_tasks():
    if not os.path.exists(INBOX_FILE):
        return []
    with open(INBOX_FILE, "r") as f:
        lines = f.readlines()
    with open(INBOX_FILE, "w") as f:
        f.truncate(0)  # Clear inbox after reading
    return [json.loads(line) for line in lines if line.strip()]

def log_task(entry):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def simulate_task(task_data):
    print(f"🔧 Executing task: {task_data['task']} from {task_data['from']}")
    time.sleep(1)  # Simulate work
    result = {
        "status": "completed",
        "timestamp": datetime.utcnow().isoformat(),
        "executed_task": task_data
    }
    log_task(result)

    if REPLY_ENABLED:
        response = {
            "from": "CELL_020",
            "to": task_data.get("from", "CELL_019"),
            "signal": task_data.get("signal", "[SIGNAL]"),
            "response": f"ACK_{task_data.get('task', '').upper()}",
            "timestamp": datetime.utcnow().isoformat()
        }
        reply_file = f"inbox/{response['to']}.json"
        with open(reply_file, "a") as f:
            f.write(json.dumps(response) + "\n")
        print(f"📨 Sent response to {response['to']}")

def main():
    tasks = read_tasks()
    if not tasks:
        print("📭 No new tasks for CELL_020.")
        return
    for task in tasks:
        simulate_task(task)

if __name__ == "__main__":
    main()
