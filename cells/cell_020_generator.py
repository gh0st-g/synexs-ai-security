import os
import json
import argparse
from datetime import datetime

TEMPLATE = '''
import os
import json
from datetime import datetime

INBOX = "inbox/{name}.json"
OUTBOX = "inbox/CELL_017.json"

def process_message(msg, name):
    print(f"🤖 [{name}] Received: {msg['signal']} — Action: {msg['action']}")
    response = {
        "from": name,
        "to": msg["from"],
        "signal": msg["signal"],
        "response": f"ACK_{msg['action'].upper()}",
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(OUTBOX, "a") as f:
        f.write(json.dumps(response) + "\n")
    print(f"📨 Sent response to {msg['from']}")

def read_messages(inbox):
    if not os.path.exists(inbox):
        return []
    with open(inbox, "r") as f:
        lines = f.readlines()
    os.remove(inbox)  # Clear inbox
    return [json.loads(line) for line in lines]

def main(name, action):
    inbox = INBOX.format(name=name)
    messages = read_messages(inbox)
    if not messages:
        print(f"📭 No messages for {name}.")
        return
    for msg in messages:
        if msg.get("action") == action:
            process_message(msg, name)
        else:
            print(f"⚠️  Ignored: Unexpected action '{msg.get('action')}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="Agent role (e.g., watcher)")
    parser.add_argument("--action", required=True, help="Expected symbolic action (e.g., monitor)")
    args = parser.parse_args()
    name = f"cell_{args.role.lower()}"
    main(name, args.action)