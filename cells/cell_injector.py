import os
import json
import random
from datetime import datetime
from typing import List, Optional, Dict

INBOX_DIR = "inbox"
RESPONSE_DIR = "response"

def load_inbox() -> List[Dict]:
    inboxes = [os.path.join(INBOX_DIR, f) for f in os.listdir(INBOX_DIR) if f.endswith(".json") and "CELL_INJECTOR" not in f]
    messages = []
    for inbox_file in inboxes:
        try:
            with open(inbox_file, "r") as f:
                messages.extend(json.load(f))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            continue
    return messages

def choose_target() -> Optional[str]:
    inboxes = [f for f in os.listdir(INBOX_DIR) if f.endswith(".json") and "CELL_INJECTOR" not in f]
    if not inboxes:
        return None
    return random.choice(inboxes)

def inject_payload(signal: str = "[SIGNAL]", reply_to: str = "CELL_INJECTOR") -> Optional[str]:
    target_file = choose_target()
    if not target_file:
        return None
    message = {
        "from": "CELL_INJECTOR",
        "to": target_file.replace(".json", ""),
        "signal": signal,
        "action": "inject",
        "timestamp": datetime.utcnow().isoformat(),
        "reply_to": reply_to
    }
    target_path = os.path.join(RESPONSE_DIR, target_file)
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "a") as f:
            f.write(json.dumps(message) + "\n")
    except (FileNotFoundError, PermissionError):
        return None
    return target_file

def respond_to(sender: str, signal: str, result: str) -> None:
    response = {
        "from": "CELL_INJECTOR",
        "to": sender,
        "signal": signal,
        "response": result,
        "timestamp": datetime.utcnow().isoformat()
    }
    inbox_file = os.path.join(RESPONSE_DIR, f"{sender}.json")
    try:
        os.makedirs(os.path.dirname(inbox_file), exist_ok=True)
        with open(inbox_file, "a") as f:
            f.write(json.dumps(response) + "\n")
    except (FileNotFoundError, PermissionError):
        print(f"⚠️ Failed to write response to {sender}'s inbox.")

def process() -> None:
    try:
        messages = load_inbox()
        if not messages:
            print("📭 No messages for CELL_INJECTOR.")
            return

        for msg in messages:
            signal = msg.get("signal", "[SIGNAL]")
            sender = msg.get("from", "UNKNOWN")
            injected_to = inject_payload(signal)
            if injected_to:
                respond_to(sender, signal, f"✅ Payload injected into {injected_to}")
            else:
                respond_to(sender, signal, "⚠️ No available target inboxes.")
    except Exception as e:
        print(f"⚠️ Error occurred: {e}")

if __name__ == "__main__":
    while True:
        try:
            process()
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error occurred: {e}")
            continue