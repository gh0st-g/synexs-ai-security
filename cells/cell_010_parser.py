import os
import json
import re
from pathlib import Path

print(">>> [cell_010] Symbolic Parser activated...\n")

INPUT_DIR = "datasets/refined"
OUTPUT_DIR = "datasets/parsed"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def parse_sequence(sequence):
    tokens = sequence.split()
    return {
        "tokens": tokens,
        "length": len(tokens),
        "keywords": [t for t in tokens if t.isupper() or t.startswith("[")],
        "directives": re.findall(r'\[(.*?)\]', sequence),
    }

def process_file(fname):
    try:
        with open(os.path.join(INPUT_DIR, fname), "r") as f:
            data = json.load(f)

        parsed = []
        for entry in data:
            if isinstance(entry, dict):
                sequence = entry.get("sequence", "")
            elif isinstance(entry, str):
                sequence = entry
            else:
                continue
            parsed.append({"original": sequence, "parsed": parse_sequence(sequence)})

        outname = f"parsed_{fname}"
        with open(os.path.join(OUTPUT_DIR, outname), "w") as f:
            json.dump(parsed, f, indent=2)

        print(f"[cell_010] Parsed {len(parsed)} entries from {fname} -> {outname}")
    except Exception as e:
        print(f"[cell_010] Error parsing {fname}: {e}")

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".json"):
        process_file(fname)