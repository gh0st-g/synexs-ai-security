import requests
from datetime import datetime

TELEGRAM_TOKEN = "8502412220:AAFtm3ifdkT7RQiOR9gy4rOVbS7iyvWeaCA"  # Your real token
CHAT_ID = "1749138955"  # Your real chat ID

def send_telegram(ip, details):
    message = f"""
*SYNEXS HACKER ALERT*
*IP:* `{ip}`
*Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
*Vulns:* `{details.get('vulns', [])}`
*Ports:* `{details.get('open_ports', [])}`
*Source:* `shodan_nmap`
    """.strip()

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
        print(f"[TELEGRAM] Alert sent for {ip}")
    except Exception as e:
        print(f"[TELEGRAM] Failed: {e}")
