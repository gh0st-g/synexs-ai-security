import json
import os

INPUT_DIR = "datasets/refined"
OUTPUT_DIR = "datasets/pattern_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_tokens(tokens):
    return {
        "token_count": len(tokens),
        "unique_tokens": len(set(tokens)),
        "repeated_tokens": len(tokens) - len(set(tokens))
    }

def main():
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(INPUT_DIR, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Accept either a list of tokens or a dict with "refined"
            if isinstance(data, list):
                tokens = data
            elif isinstance(data, dict):
                tokens = data.get("refined", "").split()
            else:
                print(f"⚠️ [cell_005] Unexpected format in: {filename}")
                continue

            stats = analyze_tokens(tokens)
            out_path = os.path.join(OUTPUT_DIR, "analyzed_" + filename)
            with open(out_path, "w") as out:
                json.dump(stats, out, indent=4)

            print(f"✅ [cell_005] Analyzed: {filename} → {stats}")
        except Exception as e:
            print(f"❌ [cell_005] Error in {filename}: {e}")

if __name__ == "__main__":
    main()

