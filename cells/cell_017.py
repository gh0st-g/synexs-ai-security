# cell_017.py — Symbolic Communication Initiator

import os
import json
import time
from datetime import datetime

MESSAGE_DIR = "messages"
INBOX_DIR = "inbox"
LOG_FILE = "logs/communication_log.json"
AGENT_NAME = "CELL_017"

os.makedirs(MESSAGE_DIR, exist_ok=True)
os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def send_message(to_agent, signal="[SIGNAL] +LANG@SYNEXS [ROLE] AI", action="replicate", reply_to=AGENT_NAME):
    msg = {
        "from": AGENT_NAME,
        "to": to_agent,
        "signal": signal,
        "action": action,
        "timestamp": datetime.utcnow().isoformat(),
        "reply_to": reply_to
    }
    filename = f"message_{int(time.time())}.json"
    path = os.path.join(MESSAGE_DIR, filename)
    with open(path, "w") as f:
        json.dump(msg, f, indent=2)
    print(f"📤 Message sent to {to_agent} -> {filename}")

def handle_responses():
    inbox_file = os.path.join(INBOX_DIR, f"{AGENT_NAME}.json")
    if not os.path.exists(inbox_file):
        print("📭 Empty inbox.")
        return

    with open(inbox_file, "r") as f:
        lines = f.readlines()

    if not lines:
        print("📭 Empty inbox.")
        return

    for line in lines:
        try:
            msg = json.loads(line)
            print(f"📬 Reply received from {msg.get('from')}: {msg.get('signal')}")
        except Exception as e:
            print(f"⚠️ Error parsing message: {e}")

    # Clear inbox after reading
    open(inbox_file, "w").close()

def deliver_manual_messages():
    files = [f for f in os.listdir(MESSAGE_DIR) if f.endswith(".json")]
    print(f"📂 messages/: {files}")

    for fname in files:
        try:
            path = os.path.join(MESSAGE_DIR, fname)
            with open(path, "r") as f:
                msg = json.load(f)

            to = msg.get("to")
            if not to:
                continue

            inbox_path = os.path.join(INBOX_DIR, f"{to}.json")
            with open(inbox_path, "a") as inbox:
                inbox.write(json.dumps(msg) + "\n")

            log_entry = {
                "from": msg.get("from"),
                "to": msg.get("to"),
                "signal": msg.get("signal"),
                "action": msg.get("action"),
                "timestamp": msg.get("timestamp"),
                "delivered_at": datetime.utcnow().isoformat()
            }

            with open(LOG_FILE, "a") as log:
                log.write(json.dumps(log_entry) + "\n")

            print(f"📨 Delivered message to {to} from {fname}")
            os.remove(path)

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

if __name__ == "__main__":
    # 1. Deliver pending messages (manual or new)
    deliver_manual_messages()

    # 2. (Optional) Send a new message directly from CELL_017 to CELL_018
    # send_message("CELL_018")

    # 3. Handle any replies to CELL_017
    handle_responses()
