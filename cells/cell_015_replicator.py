import os
import json
import random
import time
from datetime import datetime

DECISIONS_FILE = "datasets/decisions/decisions.json"
REPLICATED_DIR = "datasets/replicated"
os.makedirs(REPLICATED_DIR, exist_ok=True)

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
        print(f"⚠️  [cell_015] Could not load decisions: {e}")
        return []

def replicate_sequence(sequence, num_clones=3):
    """Create clones of sequence with variations"""
    tokens = sequence.split()
    clones = []

    for _ in range(num_clones):
        clone = tokens.copy()
        mutation_type = random.choice(["insert", "swap"])
        if mutation_type == "insert":
            index = random.randint(0, len(clone))
            clone.insert(index, "<CLONE-INJECTED>")
        elif mutation_type == "swap" and len(clone) >= 2:
            i, j = random.sample(range(len(clone)), 2)
            clone[i], clone[j] = clone[j], clone[i]
        clones.append(" ".join(clone))

    return clones

def main():
    decisions = load_decisions()
    count = 0
    sequences_to_replicate = 0

    # Filter for sequences marked for replication
    for entry in decisions:
        if not isinstance(entry, dict):
            continue

        action = entry.get("action")
        sequence = entry.get("sequence")

        if action and action.upper() == "REPLICATE" and sequence:
            sequences_to_replicate += 1
            clones = replicate_sequence(sequence)
            for clone in clones:
                data = {
                    "parent": sequence,
                    "replicated": clone,
                    "timestamp": datetime.utcnow().isoformat(),
                    "replication_type": "symbolic_clone"
                }
                filename = f"{REPLICATED_DIR}/replicated_sequence_{int(time.time()*1000)}.json"
                with open(filename, "w") as f:
                    json.dump(data, f, indent=4)
                count += 1
                time.sleep(0.001)  # Small delay to ensure unique timestamps

    print(f"✅ [cell_015] Replicated {sequences_to_replicate} sequences into {count} clones from {len(decisions)} total decisions.")

if __name__ == "__main__":
    main()
