#!/usr/bin/env python3
"""
cell_007.py - Synexs Decision Logger
This script loads decisions from JSON files, logs them to a memory log with timestamps,
and handles errors robustly. Optimized for low memory usage and append-only logging.
"""

import os
import json
import time

# Configuration Constants
DECISIONS_PATH = "datasets/decisions/decisions.json"
LOG_PATH = "datasets/memory/memory_log.json"

# Ensure directories exist
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Main Function
def main():
    """
    Main function to load decisions, create a log entry, and append to the memory log.
    Handles list or dict structures with error handling.
    """
    if not os.path.exists(DECISIONS_PATH):
        print("ℹ️ [cell_007] No decisions file found.")
        return

    try:
        with open(DECISIONS_PATH, "r") as f:
            data = json.load(f)
        # Handle different JSON structures efficiently
        decisions = data if isinstance(data, list) else data.get("decisions", [])
        if not decisions:
            print("ℹ️ [cell_007] No new decisions found.")
            return
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decisions": decisions,
            "count": len(decisions)
        }
        # Append to log file for efficiency (no full read/write)
        with open(LOG_PATH, "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
        print(f"✅ [cell_007] Logged {len(decisions)} decisions to memory.")
    except json.JSONDecodeError as e:
        print(f"❌ [cell_007] JSON decode error in decisions.json: {e}")
    except AttributeError as e:
        print(f"❌ [cell_007] Structure error in decisions.json: {e}")
    except Exception as e:
        print(f"❌ [cell_007] Error in decisions.json: {e}")

if __name__ == "__main__":
    main()
