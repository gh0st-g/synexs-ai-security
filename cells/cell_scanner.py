import os, json
from datetime import datetime

INBOX_FILE = "inbox/CELL_SCANNER.json"

def load_inbox():
    if not os.path.exists(INBOX_FILE):
        return []

    with open(INBOX_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def process():
    messages = load_inbox()
    if not messages:
        print("📭 No messages for scanner.")
        return

    for msg in messages:
        print(f"🤖 [scanner] Received: {msg.get('signal')} — Action: {msg.get('action')}")
        response = {
            "from": "scanner",
            "to": msg.get("from"),
            "signal": msg.get("signal"),
            "response": "ACK_SCAN",
            "timestamp": datetime.utcnow().isoformat()
        }

        with open(f"inbox/{response['to']}.json", "a") as f:
            f.write(json.dumps(response) + "\n")

        print(f"📨 Sent response to {response['to']}")

    open(INBOX_FILE, "w").close()

if __name__ == "__main__":
    process()
