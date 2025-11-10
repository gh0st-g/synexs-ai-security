import os
import json
import hashlib

# Directory paths
REFINED_DIR = "datasets/refined"
HASH_LOG_DIR = "datasets/hash_log"
os.makedirs(HASH_LOG_DIR, exist_ok=True)

# File hashing function
def hash_file(filepath):
    with open(filepath, "rb") as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()

# Load existing hash log
hash_log = {}
log_path = os.path.join(HASH_LOG_DIR, "hash_log.json")
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        hash_log = json.load(f)

# Ensure refined directory exists
if not os.path.exists(REFINED_DIR):
    print(f"⚠️ [cell_004] Refined directory not found: {REFINED_DIR}")
else:
    for fname in os.listdir(REFINED_DIR):
        if fname.endswith(".json") and fname not in hash_log:
            filepath = os.path.join(REFINED_DIR, fname)
            try:
                file_hash = hash_file(filepath)
                hash_log[fname] = file_hash
                print(f"✅ [cell_004] Logged hash for: {fname} — {file_hash[:8]}")
            except Exception as e:
                print(f"❌ [cell_004] Failed to hash {fname}: {e}")

    # Save updated hash log
    with open(log_path, "w") as f:
        json.dump(hash_log, f, indent=2)

print("✅ [cell_004] Hash logging complete.")

