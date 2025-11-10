import os
import json
import logging
import base64
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://mail.google.com/'
]

def get_gmail_service():
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    return build('gmail', 'v1', credentials=creds)

def get_message_detail(service, msg_id):
    return service.users().messages().get(userId='me', id=msg_id, format='full').execute()

def contains_unsubscribe_link(payload):
    try:
        parts = payload.get("parts", [])
        for part in parts:
            body = part.get("body", {})
            data = body.get("data")
            if data:
                html = base64.urlsafe_b64decode(data.encode("UTF-8")).decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                links = soup.find_all("a")
                for link in links:
                    if "unsubscribe" in link.get("href", "").lower():
                        return True
    except Exception as e:
        logging.warning(f"Unsubscribe check failed: {e}")
    return False

def delete_message(service, msg_id):
    try:
        service.users().messages().delete(userId='me', id=msg_id).execute()
        logging.info(f"🗑️ Deleted message {msg_id}")
    except Exception as e:
        logging.error(f"Failed to delete {msg_id}: {e}")

def main():
    service = get_gmail_service()
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=1000).execute()
    messages = results.get('messages', [])

    logging.info(f"📦 Found {len(messages)} messages to scan...")

    for msg in messages:
        msg_id = msg['id']
        msg_detail = get_message_detail(service, msg_id)
        payload = msg_detail.get("payload", {})
        headers = payload.get("headers", [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(No Subject)')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '(No Sender)')

        logging.info(f"Checking: {sender} — {subject}")

        if contains_unsubscribe_link(payload):
            logging.info(f"Found unsubscribe in: {sender} — Deleting...")
            delete_message(service, msg_id)
        else:
            logging.info("❌ No unsubscribe link found — Skipping.")

if __name__ == '__main__':
    main()
