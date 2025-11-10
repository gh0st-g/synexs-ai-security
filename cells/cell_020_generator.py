import os
import json
import argparse
from datetime import datetime

# ===============================
# Synexs Agent Generator — CELL_020
# ===============================

TEMPLATE = '''
import os
import json
from datetime import datetime

INBOX = "inbox/{name}.json"
OUTBOX = "inbox/CELL_017.json"

def process_message(msg):
    print(f"🤖 [{name}] Received: {{msg['signal']}} — Action: {{msg['action']}}")
    response = {{
        "from": "{name}",
        "to": msg["from"],
        "signal": msg["signal"],
        "response": "ACK_{upper_action}",
        "timestamp": datetime.utcnow().isoformat()
    }}
    with open(OUTBOX, "a") as f:
        f.write(json.dumps(response) + "\\n")
    print(f"📨 Sent response to {{msg['from']}}")

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
        print("📭 No messages for {name}.")
        return
    for msg in messages:
        if msg.get("action") == "{action}":
            process_message(msg)
        else:
            print(f"⚠️  Ignored: Unexpected action '{{msg.get('action')}}'")

if __name__ == "__main__":
    main()
'''

def generate_agent(role, action):
    name = f"cell_{role.lower()}"
    filename = f"{name}.py"
    upper_action = action.upper()
    content = TEMPLATE.format(name=name, action=action, upper_action=upper_action)
    with open(filename, "w") as f:
        f.write(content.strip())
    print(f"🛠️  Generated agent: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="Agent role (e.g., watcher)")
    parser.add_argument("--action", required=True, help="Expected symbolic action (e.g., monitor)")
    args = parser.parse_args()
    generate_agent(args.role, args.action)
