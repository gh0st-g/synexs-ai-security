import os
import json
import random
import time
from datetime import datetime

MEMORY_FILE = "memory/memory_log.json"
REPLICATED_DIR = "datasets/replicated"
os.makedirs(REPLICATED_DIR, exist_ok=True)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def replicate_sequence(sequence, num_clones=3):
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
    memory = load_memory()
    count = 0

    for entry in memory:
        if entry.get("decision") == "replicate":
            original = entry["sequence"]
            clones = replicate_sequence(original)
            for clone in clones:
                data = {
                    "parent": original,
                    "replicated": clone,
                    "timestamp": datetime.utcnow().isoformat(),
                    "replication_type": "symbolic_clone"
                }
                filename = f"{REPLICATED_DIR}/replicated_sequence_{int(time.time()*1000)}.json"
                with open(filename, "w") as f:
                    json.dump(data, f, indent=4)
                count += 1

    print(f"✅ [cell_015] Total replicated sequences saved: {count}")

if __name__ == "__main__":
    main()
