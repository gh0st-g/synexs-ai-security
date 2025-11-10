import os
import json
import random
import base64
import time
import socket
from typing import List
from pathlib import Path

# ================= CONFIG =================
MEMORY_LOG = "datasets/memory/memory_log.json"
AGENT_DIR = "datasets/agents"
LISTENER_IP = "157.245.3.180"
LISTENER_PORT = 8443
Path(AGENT_DIR).mkdir(parents=True, exist_ok=True)

# ================ LOAD DECISIONS ================
def load_decisions() -> List[str]:
    try:
        if os.path.exists(MEMORY_LOG):
            with open(MEMORY_LOG, "r") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            return ["replicate", "mutate", "self_destruct"] * 200
    except Exception as e:
        print(f"[ERROR] Reading memory: {e}")
        return ["replicate", "mutate", "self_destruct"] * 200

# ================ BUILD SCP MESSAGE ================
def build_scp(agent_id: str, decision: str) -> dict:
    return {
        "scp_version": "1.0",
        "signal_id": f"{agent_id}-{int(time.time())}",
        "origin": f"agent@{socket.gethostname()}",
        "target": "cell_012",
        "timestamp": int(time.time()),
        "symbolic_payload": "△□◆◇□□△" if decision == "replicate" else "□△◆",
        "semantic_hint": f"action:{decision}",
        "tags": [decision]
    }

# ================ GENERATE AGENT CODE ================
def generate_agent_blob(agent_id: str, decision: str) -> str:
    scp_msg = build_scp(agent_id, decision)
    scp_json = json.dumps(scp_msg, separators=(',', ':'))

    return f'''
import socket, json, base64, os, time

AGENT_ID = "{agent_id}"
DECISION = "{decision}"
LISTENER = ("{LISTENER_IP}", {LISTENER_PORT})
SCP = {scp_json}

def report():
    try:
        with socket.socket() as s:
            s.settimeout(5)
            s.connect(LISTENER)
            s.sendall(base64.b64encode(json.dumps(SCP).encode()))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to report: {e}")
        return False

if DECISION == "self_destruct":
    try:
        os._exit(0)
    except Exception as e:
        print(f"[ERROR] Failed to self-destruct: {e}")
        return

if report():
    print(f"[{agent_id}] SCP sent: {decision}")
else:
    print(f"[{agent_id}] Failed to send SCP")