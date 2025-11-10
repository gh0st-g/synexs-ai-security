# cell_017_2.py - Message Receiver & Responder

import os
import json
from datetime import datetime

INBOX_DIR = "inbox"
LOG_FILE = "logs/communication_log.json"
OUTBOX_DIR = "comm_outbox"
os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(OUTBOX_DIR, exist_ok=True)

def read_messages():
    inbox_file = os.path.join(INBOX_DIR, "CELL_017_2.json")
    if not os.path.exists(inbox_file):
        return []

    with open(inbox_file, "r") as f:
        lines = f.readlines()
    open(inbox_file, "w").close()  # clear inbox after reading
    return [json.loads(line) for line in lines]

def respond_to(message):
    return {
        "from": "CELL_017_2",
        "to": message["from"],
        "signal": message["signal"],
        "response": "ACK_REPLICATE",
        "timestamp": datetime.utcnow().isoformat()
    }

def save_response(response):
    filename = f"response_{int(datetime.utcnow().timestamp())}.json"
    with open(os.path.join(OUTBOX_DIR, filename), "w") as f:
        json.dump(response, f, indent=2)

def log(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def main():
    messages = read_messages()
    if not messages:
        print("📭 No new messages.")
        return

    for msg in messages:
        print(f"📥 Received from {msg['from']}: {msg['signal']}")
        response = respond_to(msg)
        save_response(response)
        log({
            "received": msg,
            "responded_with": response,
            "processed_at": datetime.utcnow().isoformat()
        })
        print(f"✅ Responded with: {response['response']}")

if __name__ == "__main__":
    main()
