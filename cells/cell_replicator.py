import os
import json
from datetime import datetime

INBOX_FILE = "inbox/CELL_REPLICATOR.json"
RESPONSE_DIR = "inbox"
AGENT_DIR = "agents"  # Save in agents/ directory
os.makedirs(AGENT_DIR, exist_ok=True)

TEMPLATE = '''import os, json
from datetime import datetime

INBOX_FILE = "inbox/CELL_{role_upper}.json"

def load_inbox():
    if not os.path.exists(INBOX_FILE):
        return []
    with open(INBOX_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def process():
    messages = load_inbox()
    if not messages:
        print("📭 No messages for {role}.")
        return
    for msg in messages:
        print(f"🤖 [{role}] Received: {{msg.get('signal')}} — Action: {{msg.get('action')}}")
        response = {{
            "from": "{role}",
            "to": msg.get("from"),
            "signal": msg.get("signal"),
            "response": "ACK_{action_upper}",
            "timestamp": datetime.utcnow().isoformat()
        }}
        with open(f"inbox/{{response['to']}}.json", "a") as f:
            f.write(json.dumps(response) + "\\n")
        print(f"📨 Sent response to {{response['to']}}")
    open(INBOX_FILE, "w").close()

if __name__ == "__main__":
    process()
'''

def load_inbox():
    if not os.path.exists(INBOX_FILE):
        return []
    try:
        with open(INBOX_FILE, "r") as f:
            return [json.loads(line) for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"❌ Error reading {INBOX_FILE}: {e}")
        return []

def respond_to(sender, signal, result):
    response = {
        "from": "CELL_REPLICATOR",
        "to": sender,
        "signal": signal,
        "response": result,
        "timestamp": datetime.utcnow().isoformat()
    }
    inbox_file = os.path.join(RESPONSE_DIR, f"{sender}.json")
    try:
        with open(inbox_file, "a") as f:
            f.write(json.dumps(response) + "\n")
        print(f"📨 Sent response to {sender}: {result}")
    except Exception as e:
        print(f"❌ Error writing to {inbox_file}: {e}")

def generate_agent(role="clone", action="respond"):
    filename = f"cell_{role}.py"
    content = TEMPLATE.format(role=role, action=action, role_upper=role.upper(), action_upper=action.upper())
    path = os.path.join(AGENT_DIR, filename)
    try:
        with open(path, "w") as f:
            f.write(content)
        print(f"✅ Generated agent: {path}")
        return filename
    except Exception as e:
        print(f"❌ Error generating {filename}: {e}")
        return None

def process():
    messages = load_inbox()
    if not messages:
        print("📭 No messages for CELL_REPLICATOR.")
        return
    try:
        open(INBOX_FILE, "w").close()
    except Exception as e:
        print(f"❌ Error clearing {INBOX_FILE}: {e}")
        return
    for msg in messages:
        signal = msg.get("signal", "").lower()
        sender = msg.get("from", "UNKNOWN")
        role = msg.get("target", "clone").lower()
        action = msg.get("action", "respond").lower()
        if "replicate" in signal:
            generated = generate_agent(role, action)
            if generated:
                respond_to(sender, signal, f"✅ Replicated agent: {generated}")
            else:
                respond_to(sender, signal, "❌ Failed to replicate agent.")
        else:
            respond_to(sender, signal, "❌ Unknown signal.")

if __name__ == "__main__":
    process()
