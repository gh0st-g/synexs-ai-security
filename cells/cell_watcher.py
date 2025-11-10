import os
import json
from datetime import datetime

INBOX = "inbox/cell_watcher.json"
OUTBOX = "inbox/CELL_017.json"

def process_message(msg):
    print(f"🤖 [cell_watcher] Received: {msg['signal']} — Action: {msg['action']}")
    response = {
        "from": "cell_watcher",
        "to": msg["from"],
        "signal": msg["signal"],
        "response": "ACK_MONITOR",
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(OUTBOX, "a") as f:
        f.write(json.dumps(response) + "\n")
    print(f"📨 Sent response to {msg['from']}")

def read_messages():
    if not os.path.exists(INBOX):
        return []
    with open(INBOX, "r") as f:
        lines = f.readlines()
    open(INBOX, "w").close()  # Clear inbox
    return [json.loads(line) for line in lines]

def main():
    messages = read_messages()
    if not messages:
        print("📭 No messages for cell_watcher.")
        return
    for msg in messages:
        if msg.get("action") == "monitor":
            process_message(msg)
        else:
            print(f"⚠️  Ignored: Unexpected action '{msg.get('action')}'")

if __name__ == "__main__":
    main()