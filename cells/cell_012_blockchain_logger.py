# cell_012_blockchain_logger.py

import os
import json
import hashlib
import time
from datetime import datetime

REFINED_DIR = "datasets/refined"
BLOCKCHAIN_LOG = "datasets/blockchain/blockchain_log.json"

os.makedirs("datasets/blockchain", exist_ok=True)
seen_files = set()

def sha256_of_file(filepath):
    with open(filepath, "rb") as f:
        file_data = f.read()
    return hashlib.sha256(file_data).hexdigest()

def get_last_hash():
    if not os.path.exists(BLOCKCHAIN_LOG):
        return None
    with open(BLOCKCHAIN_LOG, "r") as f:
        chain = json.load(f)
        if not chain:
            return None
        return chain[-1]["current_hash"]

def log_to_blockchain(filename, file_hash):
    prev_hash = get_last_hash()
    timestamp = datetime.utcnow().isoformat() + "Z"

    block = {
        "timestamp": timestamp,
        "filename": filename,
        "current_hash": file_hash,
        "previous_hash": prev_hash
    }

    if os.path.exists(BLOCKCHAIN_LOG):
        with open(BLOCKCHAIN_LOG, "r") as f:
            chain = json.load(f)
    else:
        chain = []

    chain.append(block)

    with open(BLOCKCHAIN_LOG, "w") as f:
        json.dump(chain, f, indent=4)

    print(f"🧾 [cell_012] Logged {filename} with hash {file_hash[:8]}...")

def main():
    print("🔒 [cell_012] Blockchain logger started...")
    os.makedirs(REFINED_DIR, exist_ok=True)

    while True:
        try:
            for fname in os.listdir(REFINED_DIR):
                if not fname.endswith(".json") or fname in seen_files:
                    continue
                path = os.path.join(REFINED_DIR, fname)
                file_hash = sha256_of_file(path)
                log_to_blockchain(fname, file_hash)
                seen_files.add(fname)
        except Exception as e:
            print(f"❌ [cell_012] Error: {e}")

        time.sleep(10)  # Wait 10 seconds before scanning again

if __name__ == "__main__":
    main()
