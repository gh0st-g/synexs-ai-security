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
            "banner": host.get("data", [{}])[0].get("data", "")
        }
    except shodan.APIError as e:
        print(f"[Shodan] Error for {ip}: {e}")
        return {}
    except Exception as e:
        print(f"[Shodan] Unexpected error for {ip}: {e}")
        return {}

def nmap_scan(ip: str, quick=True) -> dict:
    nm = nmap.PortScanner()
    args = "-F -sV" if quick else "-sV --script vuln -T4"
    try:
        nm.scan(ip, arguments=args, sudo=True)
        if ip in nm.all_hosts():
            return nm[ip].get("tcp", {})
        else:
            return {}
    except nmap.nmap.PortScannerError as e:
        print(f"[Nmap] Error for {ip}: {e}")
        return {}
    except Exception as e:
        print(f"[Nmap] Unexpected error for {ip}: {e}")
        return {}