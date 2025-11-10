import json
import os
from datetime import datetime

EMAIL_FILE = "emails_tagged.json"
LOG_FILE = "cleanup_log.json"

def load_emails():
    if os.path.exists(EMAIL_FILE):
        with open(EMAIL_FILE, "r") as f:
            return json.load(f)
    return []

def save_emails(data):
    with open(EMAIL_FILE, "w") as f:
        json.dump(data, f, indent=2)

def log_cleanup(entry):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def auto_cleanup():
    emails = load_emails()
    kept_emails = []
    deleted = 0
    unsubscribed = 0

    for email in emails:
        action = email.get("action", "").lower()
        if action == "delete":
            log_cleanup({"time": str(datetime.now()), "action": "delete", "email": email})
            deleted += 1
        elif action == "unsubscribe":
            log_cleanup({"time": str(datetime.now()), "action": "unsubscribe", "email": email})
            unsubscribed += 1
        else:
            kept_emails.append(email)

    save_emails(kept_emails)
    print(f"✅ Cleanup complete: {deleted} deleted, {unsubscribed} unsubscribed, {len(kept_emails)} kept.")

if __name__ == "__main__":
    auto_cleanup()
