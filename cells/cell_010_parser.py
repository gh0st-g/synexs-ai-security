import os
import json
import re

print(">>> [cell_010] Symbolic Parser activated...\n")

input_dir = "datasets/refined"
output_dir = "datasets/parsed"
os.makedirs(output_dir, exist_ok=True)

def parse_sequence(sequence):
    return {
        "tokens": sequence.split(),
        "length": len(sequence.split()),
        "keywords": [t for t in sequence.split() if t.isupper() or t.startswith("[")],
        "directives": re.findall(r'\[(.*?)\]', sequence),
    }

for fname in os.listdir(input_dir):
    if not fname.endswith(".json"):
        continue

    try:
        with open(os.path.join(input_dir, fname), "r") as f:
            data = json.load(f)

        parsed = []

        for entry in data:
            if isinstance(entry, dict):
                sequence = entry.get("sequence", "")
            elif isinstance(entry, str):
                sequence = entry
            else:
                continue

            parsed.append({
                "original": sequence,
                "parsed": parse_sequence(sequence)
            })

        outname = f"parsed_{fname}"
        with open(os.path.join(output_dir, outname), "w") as f:
            json.dump(parsed, f, indent=2)

        print(f"[cell_010] Parsed {len(parsed)} entries from {fname} -> {outname}")

    except Exception as e:
        print(f"[cell_010] Error parsing {fname}: {e}")
