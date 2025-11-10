import os
import json
from datetime import datetime
from pathlib import Path
from time import sleep

INBOX = Path("inbox", "cell_analyzer.json")
OUTBOX = Path("inbox", "CELL_017.json")

def process_message(msg):
    print(f"🤖 [cell_analyzer] Received: {msg['signal']} — Action: {msg['action']}")
    response = {
        "from": "cell_analyzer",
        "to": msg["from"],
        "signal": msg["signal"],
        "response": "ACK_DETECT",
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        with OUTBOX.open("a") as f:
            f.write(json.dumps(response) + "\n")
    except (IOError, OSError) as e:
        print(f"⚠️  Error writing to outbox: {e}")
    else:
        print(f"📨 Sent response to {msg['from']}")

def read_messages():
    messages = []
    if INBOX.exists():
        try:
            with INBOX.open("r") as f:
                messages = [json.loads(line) for line in f.readlines()]
        except (IOError, OSError) as e:
            print(f"⚠️  Error reading from inbox: {e}")
        finally:
            try:
                INBOX.unlink(missing_ok=True)  # Clear inbox
            except (IOError, OSError) as e:
                print(f"⚠️  Error clearing inbox: {e}")
    return messages

def main():
    while True:
        try:
            messages = read_messages()
            if not messages:
                print("📭 No messages for cell_analyzer.")
                sleep(1)  # Add a delay to prevent excessive CPU usage
                continue

            for msg in messages:
                if msg.get("action") == "detect":
                    process_message(msg)
                else:
                    print(f"⚠️  Ignored: Unexpected action '{msg.get('action')}'")
        except Exception as e:
            print(f"⚠️  Error processing messages: {e}")
            sleep(1)  # Add a delay to prevent excessive CPU usage

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting cell_analyzer...")
    except Exception as e:
        print(f"⚠️  Unhandled error: {e}")