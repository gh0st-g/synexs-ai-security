import shodan
import nmap
import requests
from datetime import datetime

SHODAN_API_KEY = "SsRKirkAqOloEve6f75r67HwrBXcOjlN"

def shodan_lookup(ip: str) -> dict:
    try:
        api = shodan.Shodan(SHODAN_API_KEY)
        host = api.host(ip)
        return {
            "open_ports": host.get("ports", []),
            "vulns": [v["cve"] for v in host.get("vulns", [])],
            "banner": host.get("data", [{}])[0].get("data", ""),
            "country": host.get("country_name"),
            "country_code": host.get("country_code"),
            "city": host.get("city"),
            "latitude": host.get("latitude"),
            "longitude": host.get("longitude"),
            "org": host.get("org"),
            "isp": host.get("isp"),
            "asn": host.get("asn")
        }
    except Exception as e:
        print(f"[Shodan] Error for {ip}: {e}")
        return {}

def nmap_scan(ip: str, quick=True) -> dict:
    nm = nmap.PortScanner()
    args = "-F -sV" if quick else "-sV --script vuln -T4"
    try:
        nm.scan(ip, arguments=args)
        return nm[ip].get("tcp", {}) if ip in nm.all_hosts() else {}
    except Exception as e:
        print(f"[Nmap] Error for {ip}: {e}")
        return {}
