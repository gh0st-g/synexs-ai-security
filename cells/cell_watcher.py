import os
import json
from datetime import datetime
from pathlib import Path

INBOX = Path("inbox/cell_watcher.json")
OUTBOX = Path("inbox/CELL_017.json")

def process_message(msg):
    print(f"🤖 [cell_watcher] Received: {msg['signal']} — Action: {msg['action']}")
    response = {
        "from": "cell_watcher",
        "to": msg["from"],
        "signal": msg["signal"],
        "response": "ACK_MONITOR",
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        with open(OUTBOX, "a") as f:
            f.write(json.dumps(response) + "\n")
    except OSError as e:
        print(f"⚠️ Error writing to outbox: {e}")
    else:
        print(f"📨 Sent response to {msg['from']}")

def read_messages():
    try:
        if not INBOX.exists():
            return []
        with open(INBOX, "r") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"⚠️ Error reading from inbox: {e}")
        return []
    else:
        INBOX.unlink(missing_ok=True)  # Clear inbox
        return [json.loads(line) for line in lines]

def main():
    try:
        messages = read_messages()
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return
    if not messages:
        print("📭 No messages for cell_watcher.")
        return
    for msg in messages:
        if msg.get("action") == "monitor":
            try:
                process_message(msg)
            except Exception as e:
                print(f"⚠️ Error processing message: {e}")
        else:
            print(f"⚠️  Ignored: Unexpected action '{msg.get('action')}'")

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
        finally:
            # Add a delay to avoid busy-waiting
            import time
            time.sleep(5)