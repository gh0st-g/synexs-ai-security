import json
import os
from datetime import datetime
from pathlib import Path

INSTINCT_LOG = "logs/instinct_log.json"
AGENT_OUTPUT_DIR = "generated_agents"

Path(AGENT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(INSTINCT_LOG)).mkdir(parents=True, exist_ok=True)

def generate_agent_code(goal):
    if "EMAIL_AGENT" in goal:
        return """# auto_generated_email_agent.py
import imaplib
import email

def check_inbox():
    print("📧 Scanning inbox for important emails...")
    # TODO: Connect via IMAP, scan for filters

if __name__ == "__main__":
    check_inbox()
"""
    return "# Unknown agent goal"

def instinct_think(signal):
    timestamp = datetime.utcnow().isoformat()
    agent_type = "EMAIL_AGENT" if "EMAIL_AGENT" in signal else "UNKNOWN"
    filename = f"agent_{agent_type.lower()}_{int(datetime.utcnow().timestamp())}.py"
    filepath = os.path.join(AGENT_OUTPUT_DIR, filename)

    try:
        with open(filepath, "w") as f:
            f.write(generate_agent_code(signal))
    except Exception as e:
        print(f"Error generating agent: {e}")
        return None

    log_entry = {
        "input_signal": signal,
        "generated_agent": filename,
        "timestamp": timestamp
    }

    try:
        with open(INSTINCT_LOG, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(log_entry)

    try:
        with open(INSTINCT_LOG, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error logging agent: {e}")

    print(f"✅ Generated agent: {filename}")
    return filepath

def main():
    while True:
        try:
            example_signal = "[SIGNAL] +LANG@SYNEXS [ACTION] DEPLOY:EMAIL_AGENT"
            instinct_think(example_signal)
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

if __name__ == "__main__":
    main()