import json
import os
import random
import time

DECISIONS_FILE = "datasets/decisions/decisions.json"
MUTATED_DIR = "datasets/mutated"
os.makedirs(MUTATED_DIR, exist_ok=True)

def load_decisions():
    """Load decisions from cell_006 classification output"""
    try:
        with open(DECISIONS_FILE, "r") as f:
            data = json.load(f)
            # Handle both formats: {"decisions": [...]} or [...]
            if isinstance(data, dict):
                decisions = data.get("decisions", [])
            elif isinstance(data, list):
                decisions = data
            else:
                decisions = []
            if not isinstance(decisions, list):
                return []
            return decisions
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  [cell_014] Could not load decisions: {e}")
        return []

def mutate_sequence(seq):
    """Apply mutation to sequence (token reversal)"""
    tokens = seq.split()
    if len(tokens) > 1:
        idx = random.randint(0, len(tokens) - 1)
        tokens[idx] = tokens[idx][::-1]  # Reverse token
    return " ".join(tokens)

def main():
    decisions = load_decisions()
    mutated_count = 0

    # Filter for sequences marked for mutation
    for entry in decisions:
        if not isinstance(entry, dict):
            continue

        action = entry.get("action")
        sequence = entry.get("sequence")

        if action and action.upper() == "MUTATE" and sequence:
            mutated = mutate_sequence(sequence)
            timestamp = int(time.time() * 1000)  # Use milliseconds for unique filenames
            filename = f"{MUTATED_DIR}/mutated_sequence_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump({
                    "original": sequence,
                    "mutated": mutated,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=4)
            mutated_count += 1
            time.sleep(0.001)  # Small delay to ensure unique timestamps

    print(f"✅ [cell_014] Mutated {mutated_count} sequences from {len(decisions)} total decisions.")

if __name__ == "__main__":
    main()
