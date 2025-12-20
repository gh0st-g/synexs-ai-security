import os
import json
import hashlib
from pathlib import Path

# Directory paths
REFINED_DIR = Path("datasets/refined")
HASH_LOG_DIR = Path("datasets/hash_log")
HASH_LOG_PATH = HASH_LOG_DIR / "hash_log.json"

# Ensure directories exist
HASH_LOG_DIR.mkdir(parents=True, exist_ok=True)

# File hashing function
def hash_file(filepath: Path) -> str:
    with filepath.open("rb") as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()

# Load existing hash log
hash_log = {}
if HASH_LOG_PATH.exists():
    with HASH_LOG_PATH.open("r") as f:
        hash_log = json.load(f)

# Process files in the refined directory
for fname in REFINED_DIR.glob("*.json"):
    if fname.name not in hash_log:
        try:
            file_hash = hash_file(fname)
            hash_log[fname.name] = file_hash
            print(f"✅ Logged hash for: {fname.name} — {file_hash[:8]}")
        except Exception as e:
            print(f"❌ Failed to hash {fname.name}: {e}")

# Save updated hash log
with HASH_LOG_PATH.open("w") as f:
    json.dump(hash_log, f, indent=2)

print("✅ Hash logging complete.")