#!/usr/bin/env python3
"""
cell_013_memory_logger.py - Synexs Memory Logger
This script logs classified sequences and decisions from decisions.json and routed files
to a memory log with timestamps. Optimized for low memory usage with append-only logging
and efficient data handling on a resource-constrained VPS.
"""

import os
import json
import time

# Configuration Constants
DECISIONS_PATH = "datasets/decisions/decisions.json"
ROUTED_DIR = "datasets/routed"
MEMORY_LOG_PATH = "datasets/memory/memory_log.json"

# Ensure directory exists (optimized to avoid redundant checks)
os.makedirs(os.path.dirname(MEMORY_LOG_PATH), exist_ok=True)

# Main Function
def main():
    """
    Main function to load decisions and sequences, create log entries, and append to memory log.
    Handles various JSON structures with minimal memory overhead and robust error handling.
    """
    sequences = []

    # Load decisions from decisions.json
    if os.path.exists(DECISIONS_PATH):
        try:
            with open(DECISIONS_PATH, "r") as f:
                data = json.load(f)
            decisions = data if isinstance(data, list) else data.get("decisions", [])
            for item in decisions:
                if isinstance(item, dict):
                    seq = item.get("sequence", "")
                elif isinstance(item, str):
                    seq = item
                if seq:
                    sequences.append({"sequence": seq})
            print(f"✅ [cell_013] Loaded {len(decisions)} decisions from {DECISIONS_PATH}")
        except json.JSONDecodeError as e:
            print(f"❌ [cell_013] JSON decode error in {DECISIONS_PATH}: {e}")
        except Exception as e:
            print(f"❌ [cell_013] Error processing {DECISIONS_PATH}: {e}")

    # Load sequences from routed files (optimized with early exit if no files)
    routed_files = [f for f in os.listdir(ROUTED_DIR) if f.endswith(".json")]
    if routed_files:
        for fname in routed_files:
            path = os.path.join(ROUTED_DIR, fname)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    seq = data.get("sequence", "")
                elif isinstance(data, str):
                    seq = data
                if seq:
                    sequences.append({"sequence": seq})
                print(f"✅ [cell_013] Loaded sequence from {fname}")
            except json.JSONDecodeError as e:
                print(f"❌ [cell_013] JSON decode error in {fname}: {e}")
            except Exception as e:
                print(f"❌ [cell_013] Error processing {fname}: {e}")

    # Log sequences if any (optimized append-only write)
    if sequences:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sequences": sequences,
            "count": len(sequences)
        }
        try:
            with open(MEMORY_LOG_PATH, "a") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
            print(f"✅ [cell_013] Logged {len(sequences)} sequences to {MEMORY_LOG_PATH}")
        except Exception as e:
            print(f"❌ [cell_013] Error writing to {MEMORY_LOG_PATH}: {e}")
    else:
        print("ℹ️ [cell_013] No new sequences found.")

if __name__ == "__main__":
    main()
