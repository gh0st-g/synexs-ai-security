import json
from suggest_action import suggest_action
from pathlib import Path
import logging
import time

INPUT_FILE = "emails_tagged.jsonl"
OUTPUT_FILE = "emails_tagged.jsonl"

def load_emails():
    try:
        emails = []
        with open(INPUT_FILE, "r") as f:
            for line in f:
                try:
                    email = json.loads(line.strip())
                    emails.append(email)
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON: {line.strip()}")
        return emails
    except FileNotFoundError:
        logging.error(f"File not found: {INPUT_FILE}")
        return []
    except IOError as e:
        logging.error(f"Error loading emails: {e}")
        return []

def update_actions(emails):
    return [{"action": suggest_action(email), **email} for email in emails]

def save_emails(emails):
    try:
        with open(OUTPUT_FILE, "w") as f:
            for email in emails:
                f.write(json.dumps(email) + "\n")
    except IOError as e:
        logging.error(f"Error saving emails: {e}")

def main():
    while True:
        try:
            emails = load_emails()
            if emails:
                updated_emails = update_actions(emails)
                save_emails(updated_emails)
                logging.info(f"✅ Updated actions for {len(updated_emails)} emails.")
            else:
                logging.info("No emails found.")
        except Exception as e:
            logging.error(f"Error: {e}")
        time.sleep(3600)  # Wait for 1 hour before checking again

if __name__ == "__main__":
    logging.basicConfig(filename='update_actions.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main()