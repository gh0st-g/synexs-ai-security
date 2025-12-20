import json
import os
import random
import time
from pathlib import Path

DECISIONS_FILE = "datasets/decisions/decisions.json"
MUTATED_DIR = "datasets/mutated"
Path(MUTATED_DIR).mkdir(parents=True, exist_ok=True)

def load_decisions():
    """Load decisions from cell_006 classification output"""
    try:
        with open(DECISIONS_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                decisions = data.get("decisions", [])
            elif isinstance(data, list):
                decisions = data
            else:
                decisions = []
        return [d for d in decisions if isinstance(d, dict)]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def mutate_sequence(seq):
    """Apply mutation to sequence (token reversal)"""
    tokens = seq.split()
    if len(tokens) > 1:
        idx = random.randint(0, len(tokens) - 1)
        tokens[idx] = tokens[idx][::-1]
    return " ".join(tokens)

def save_mutated(original, mutated, timestamp):
    filename = f"{MUTATED_DIR}/mutated_sequence_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "original": original,
            "mutated": mutated,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4)

def main():
    decisions = load_decisions()
    mutated_count = 0

    for entry in decisions:
        action = entry.get("action")
        sequence = entry.get("sequence")
        if action and action.upper() == "MUTATE" and sequence:
            mutated = mutate_sequence(sequence)
            timestamp = int(time.time() * 1000)
            save_mutated(sequence, mutated, timestamp)
            mutated_count += 1
            time.sleep(0.001)

    print(f"✅ [cell_014] Mutated {mutated_count} sequences from {len(decisions)} total decisions.")

if __name__ == "__main__":
    main()