
# Auto-generated Synexs Agent
# Role: logger

import os
import json
from datetime import datetime

INBOX_FILE = "inbox/CELL_LOGGER.json"
LOG_FILE = "logs/logger_log.json"

def main():
    if not os.path.exists(INBOX_FILE):
        print("📭 No messages for logger.")
        return
    with open(INBOX_FILE, "r") as f:
        lines = f.readlines()
    if not lines:
        print("📭 No messages for logger.")
        return

    for line in lines:
        try:
            msg = json.loads(line.strip())
            print(f"🤖 [logger] Received: {msg.get('signal')} — Action: {msg.get('action')}")
            response = {
                "from": "logger",
                "to": msg.get("from"),
                "signal": msg.get("signal"),
                "response": "ACK_" + msg.get("action", "unknown").upper(),
                "timestamp": datetime.utcnow().isoformat()
            }
            # Save response
            inbox_reply = f"inbox/{response['to']}.json"
            with open(inbox_reply, "a") as f_out:
                f_out.write(json.dumps(response) + "\n")
            print(f"📨 Sent response to {response['to']}")
        except Exception as e:
            print(f"❌ Failed to process line: {e}")

    os.remove(INBOX_FILE)

if __name__ == "__main__":
    main()
