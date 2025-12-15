import re
from typing import Dict, List, Tuple
from logging import basicConfig, getLogger, DEBUG, INFO

basicConfig(level=DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = getLogger(__name__)

def suggest_action(email: Dict[str, str]) -> str:
    try:
        tags: List[str] = email.get("tags", [])
        sender: str = email.get("sender", "").lower()
        subject: str = email.get("subject", "").lower()
        snippet: str = email.get("snippet", "").lower()
    except (KeyError, TypeError) as e:
        logger.error(f"Error processing email: {e}")
        return "⚠️ Review"

    action_map: Dict[Tuple[str, ...], str] = {
        ("finance", "personal", "work"): "Keep",
        ("promo", "ads", "sale", "newsletter"): "Unsubscribe",
        ("scam", "phishing"): "⚠️ Review" if "security alert" in subject else "Delete?",
        (): "⚠️ Review" if sender == "unknown" or any(phrase in snippet for phrase in ("verify account", "suspicious login")) else "Archive" if any(phrase in subject or snippet for phrase in ("auto-reply", "vacation")) else "Delete?"
    }

    for tags_key, action in action_map.items():
        if all(tag in tags for tag in tags_key):
            logger.info(f"Suggested action: {action}")
            return action

    logger.info("Suggested action: Delete?")
    return "Delete?"

def main():
    try:
        while True:
            email = {
                "tags": ["finance", "personal"],
                "sender": "john@example.com",
                "subject": "Your monthly statement",
                "snippet": "Here is your monthly financial statement."
            }
            action = suggest_action(email)
            print(f"Suggested action: {action}")
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()