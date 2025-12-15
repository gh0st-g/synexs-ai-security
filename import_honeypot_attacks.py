#!/usr/bin/env python3
"""
Import honeypot attacks from JSON log into PostgreSQL database
With GeoIP enrichment for latitude/longitude
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import logging

sys.path.insert(0, '/root/synexs')
from db.database import get_db
from db.models import Attack

# GeoIP imports
try:
    from geolite2 import geolite2
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False
    logging.warning("GeoIP library not available. Install with: pip3 install maxminddb-geolite2")

HONEYPOT_LOG = "/root/synexs/datasets/honeypot/attacks.json"
STATE_FILE = "/root/synexs/.honeypot_import_state.json"


def load_state():
    """Load the last processed line number"""
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_line": 0}


def save_state(state):
    """Save the current processing state"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def get_geolocation(ip: str):
    """Get geolocation data for an IP address"""
    if not GEOIP_AVAILABLE:
        return None, None, None, None, None

    # Skip private/local IPs
    if ip.startswith(('127.', '10.', '192.168.', '172.16.', '172.17.', '172.18.', '172.19.', 'localhost', 'unknown')):
        return None, None, None, None, None

    try:
        reader = geolite2.reader()
        match = reader.get(ip)

        if match and 'location' in match:
            location = match['location']
            country_data = match.get('country', {})
            city_data = match.get('city', {})

            return (
                location.get('latitude'),
                location.get('longitude'),
                country_data.get('names', {}).get('en'),
                country_data.get('iso_code'),
                city_data.get('names', {}).get('en')
            )
    except Exception as e:
        logging.error(f"GeoIP lookup error for {ip}: {e}")

    return None, None, None, None, None


def import_attacks():
    """Import new honeypot attacks into database with GeoIP enrichment"""
    if not Path(HONEYPOT_LOG).exists():
        logging.error(f"Honeypot log not found: {HONEYPOT_LOG}")
        return 0

    state = load_state()
    last_line = state.get("last_line", 0)

    with open(HONEYPOT_LOG) as f:
        lines = f.readlines()

    new_lines = lines[last_line:]
    imported = 0

    logging.info(f"Found {len(new_lines)} new attacks to import")
    if GEOIP_AVAILABLE:
        logging.info("✓ GeoIP enrichment enabled")

    with get_db() as db:
        for i, line in enumerate(new_lines):
            try:
                data = json.loads(line.strip())

                # Parse timestamp
                timestamp_str = data.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()

                # Extract vulnerabilities/threats
                vulns = []
                if "patterns" in data:
                    for pattern_type, detected in data["patterns"].items():
                        if detected:
                            vulns.append(pattern_type)

                if "detection" in data and "threats" in data["detection"]:
                    vulns.extend(data["detection"]["threats"])

                if "fake_crawler" in data and data["fake_crawler"].get("is_fake"):
                    vulns.append("fake_crawler")

                # Determine severity
                result = data.get("result", "unknown")
                if "blocked" in result or "waf" in result:
                    severity = "high" if len(vulns) > 1 else "medium"
                else:
                    severity = "low"

                # Get geolocation data
                ip_addr = data.get("ip", "unknown")
                lat, lon, country, country_code, city = get_geolocation(ip_addr)

                # Create instruction/output for training format
                endpoint = data.get("endpoint", "/")
                user_agent = data.get("user_agent", "unknown")
                instruction = f"Honeypot attack on {endpoint} from IP {ip_addr}"
                if country:
                    instruction += f" ({country})"
                output = f"Detected: {', '.join(vulns) if vulns else 'suspicious activity'} | Result: {result}"

                # Create attack record
                attack = Attack(
                    ip=ip_addr,
                    timestamp=timestamp,
                    vulns=vulns if vulns else None,
                    country=country,
                    country_code=country_code,
                    city=city,
                    latitude=lat,
                    longitude=lon,
                    severity=severity,
                    source="honeypot",
                    is_threat=(severity != "low"),
                    instruction=instruction,
                    output=output
                )

                db.add(attack)
                imported += 1

                if imported % 10 == 0:
                    logging.info(f"Imported {imported} attacks...")

            except Exception as e:
                logging.error(f"Error importing attack on line {last_line + i}: {e}")
                continue

        db.commit()

    # Update state
    state["last_line"] = last_line + len(new_lines)
    save_state(state)

    logging.info(f"✅ Successfully imported {imported} attacks into database")
    return imported


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("/var/log/honeypot_import.log"),
            logging.StreamHandler()
        ]
    )

    try:
        count = import_attacks()
        sys.exit(0 if count >= 0 else 1)
    except Exception as e:
        logging.error(f"❌ Import failed: {e}")
        sys.exit(1)