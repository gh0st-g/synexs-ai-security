#!/usr/bin/env python3
"""
Add geolocation data to existing attacks in database
"""

import sys
sys.path.insert(0, '/root/synexs')

from db.database import get_db
from db.models import Attack

try:
    from geolite2 import geolite2
    GEOIP_AVAILABLE = True
except ImportError:
    print("Error: GeoIP library not available")
    sys.exit(1)


def get_geolocation(ip: str):
    """Get geolocation data for an IP address"""
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
        print(f"GeoIP lookup error for {ip}: {e}")

    return None, None, None, None, None


def enrich_attacks():
    """Add geolocation to existing attacks"""
    with get_db() as db:
        # Get all attacks without geolocation
        attacks = db.query(Attack).filter(Attack.latitude.is_(None)).all()

        print(f"Found {len(attacks)} attacks without geolocation")
        updated = 0

        for attack in attacks:
            lat, lon, country, country_code, city = get_geolocation(attack.ip)

            if lat is not None and lon is not None:
                attack.latitude = lat
                attack.longitude = lon
                attack.country = country
                attack.country_code = country_code
                attack.city = city
                updated += 1

                if updated % 10 == 0:
                    print(f"Updated {updated} attacks...")

        db.commit()
        print(f"✅ Updated {updated} attacks with geolocation data")

        # Show summary
        total_with_geo = db.query(Attack).filter(Attack.latitude.isnot(None)).count()
        print(f"📍 Total attacks with geolocation: {total_with_geo}")


if __name__ == "__main__":
    enrich_attacks()
