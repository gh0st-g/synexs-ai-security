import os
import json
from datetime import datetime

print(">>> [cell_011] Symbolic Router activated...\n")

input_dir = "datasets/parsed"
output_dir = "datasets/routed"
os.makedirs(output_dir, exist_ok=True)

def route_packet(entry, source_file, index):
    try:
        message = {
            "origin": "cell_011",
            "timestamp": datetime.utcnow().isoformat(),
            "token_count": entry.get("length", 0),
            "keywords": entry.get("keywords", []),
            "directives": entry.get("directives", []),
            "raw": entry
        }

        routed_filename = source_file.replace("parsed_", f"routed_{index}_")
        output_path = os.path.join(output_dir, routed_filename)
        with open(output_path, "w") as f:
            json.dump(message, f, indent=2)
        print(f"[cell_011] Routed entry {index} from {source_file} to {output_path}")
    except Exception as e:
        print(f"[cell_011] Routing error on entry {index}: {e}")

# Router engine
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        try:
            with open(input_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    for i, entry in enumerate(data):
                        route_packet(entry, filename, i)
                elif isinstance(data, dict):
                    route_packet(data, filename, 0)
                else:
                    print(f"[cell_011] Unsupported format in {filename}")
        except Exception as e:
            print(f"[cell_011] Failed to route {filename}: {e}")
