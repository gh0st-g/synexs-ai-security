import json
import os
import random
import time

MEMORY_FILE = "memory/memory_log.json"
MUTATED_DIR = "datasets/mutated"
os.makedirs(MUTATED_DIR, exist_ok=True)

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
            if not isinstance(memory, list):
                return []
            return memory
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def mutate_sequence(seq):
    tokens = seq.split()
    if len(tokens) > 1:
        idx = random.randint(0, len(tokens) - 1)
        tokens[idx] = tokens[idx][::-1]  # Reverse token
    return " ".join(tokens)

def main():
    memory = load_memory()
    mutated_count = 0

    for entry in memory:
        sequence = entry.get("sequence")
        if sequence:
            mutated = mutate_sequence(sequence)
            timestamp = int(time.time())
            filename = f"{MUTATED_DIR}/mutated_sequence_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump({"original": sequence, "mutated": mutated}, f, indent=4)
            mutated_count += 1

    print(f"✅ [cell_014] Mutated {mutated_count} sequences.")

if __name__ == "__main__":
    main()
