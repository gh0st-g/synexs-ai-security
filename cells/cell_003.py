import os
import json

GENERATED_DIR = "datasets/generated"
TO_REFINE_DIR = "datasets/to_refine"

os.makedirs(TO_REFINE_DIR, exist_ok=True)

def refine_sequence(sequence_str):
    tokens = sequence_str.split()
    return [token for token in tokens if token and isinstance(token, str)]

def main():
    files = os.listdir(GENERATED_DIR)
    for fname in files:
        if not fname.endswith(".json"):
            continue
        path = os.path.join(GENERATED_DIR, fname)
        with open(path, "r") as f:
            try:
                data = json.load(f)
                sequences = data.get("sequences", [])
                refined_output = []

                for entry in sequences:
                    raw_seq = entry.get("sequence", "")
                    refined_tokens = refine_sequence(raw_seq)
                    refined_output.append({
                        "original": raw_seq,
                        "refined": " ".join(refined_tokens),
                        "purpose": entry.get("purpose", "unspecified")
                    })

                output_path = os.path.join(TO_REFINE_DIR, f"refined_{fname}")
                with open(output_path, "w") as out:
                    json.dump(refined_output, out, indent=4)
                print(f"✅ [cell_003] Refined and saved: {output_path}")
            except Exception as e:
                print(f"❌ [cell_003] Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
