import re
from typing import Dict, List, Tuple

def suggest_action(email: Dict[str, str]) -> str:
    try:
        tags: List[str] = email.get("tags", [])
        sender: str = email.get("sender", "").lower()
        subject: str = email.get("subject", "").lower()
        snippet: str = email.get("snippet", "").lower()
    except (KeyError, TypeError):
        return "⚠️ Review"

    action_map: Dict[Tuple[str, ...], str] = {
        ("finance", "personal", "work"): "Keep",
        ("promo", "ads", "sale", "newsletter"): "Unsubscribe",
        ("scam", "phishing"): "⚠️ Review" if "security alert" in subject else "Delete?",
        (): "⚠️ Review" if sender == "unknown" or any(phrase in snippet for phrase in ("verify account", "suspicious login")) else "Archive" if any(phrase in subject or snippet for phrase in ("auto-reply", "vacation")) else "Delete?"
    }

    for tags_key, action in action_map.items():
        if all(tag in tags for tag in tags_key):
            return action

    return "Delete?"