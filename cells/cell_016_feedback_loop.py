import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

MUTATED_DIR = "datasets/mutated"
REPLICATED_DIR = "datasets/replicated"
DECISIONS_PATH = "datasets/decisions/decisions.json"

def load_sequences_from_folder(folder: str) -> List[Dict]:
    sequences = []
    folder_path = Path(folder)
    for file_path in folder_path.glob("*.json"):
        try:
            with file_path.open("r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    sequences.extend(data)
                elif isinstance(data, dict):
                    sequences.append(data)
        except (json.JSONDecodeError, IOError, Exception) as e:
            print(f"❌ Error reading {file_path.name}: {e}")
    return sequences

def analyze_symbolic_structure(text: str) -> float:
    tokens = text.split()
    score = len(set(tokens)) / (len(tokens) + 1)
    return score

def save_decisions(sequences: List[Dict], origin_label: str) -> None:
    results = []
    for entry in sequences:
        text = next((value for key, value in entry.items() if key in ["mutated", "replicated", "sequence", "parent"]), None)
        if text is None:
            continue

        score = analyze_symbolic_structure(text)
        decision = "replicate" if score > 0.5 else "mutate" if score > 0.3 else "discard"
        results.append({
            "sequence": text,
            "decision": decision,
            "origin": origin_label,
            "score": round(score, 3),
            "timestamp": datetime.utcnow().isoformat()
        })

    decisions_dir = Path("datasets/decisions")
    decisions_dir.mkdir(parents=True, exist_ok=True)
    with (decisions_dir / "decisions.json").open("w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ [cell_016] {len(results)} decisions written to {DECISIONS_PATH}")

def main() -> None:
    try:
        mutated = load_sequences_from_folder(MUTATED_DIR)
        replicated = load_sequences_from_folder(REPLICATED_DIR)
        print(f"[cell_016] Loaded {len(mutated)} mutated and {len(replicated)} replicated sequences.")

        save_decisions(mutated, "mutated")
        save_decisions(replicated, "replicated")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()