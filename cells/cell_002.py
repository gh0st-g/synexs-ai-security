import os
import json
import time
import random
from cell_001 import generate_symbolic_sequence, model, vocab as imported_vocab

os.makedirs("datasets/generated", exist_ok=True)

vocab = imported_vocab or ["SIGMA", "DELTA", "THETA", "ALPHA", "ZETA", "OMEGA"]

while True:
    try:
        num_sequences = 50
        generated_sequences = [
            {"sequence": " ".join(generate_symbolic_sequence(model, random.choice(vocab))), "purpose": "AI-to-AI optimized communication"}
            for _ in range(num_sequences)
        ]

        timestamp = int(time.time())
        final_filename = f"datasets/generated/generated_scp_dataset_{timestamp}.json"

        with open(final_filename, "w") as f:
            json.dump({"sequences": generated_sequences}, f, indent=4)
        print(f"✅ [cell_002] Saved {num_sequences} new sequences to {final_filename}")

        time.sleep(3600)  # Wait for 1 hour before generating new sequences
    except (OSError, IOError, ValueError, TypeError) as e:
        print(f"❌ [cell_002] Error saving sequences to {final_filename}: {e}")
        time.sleep(60)  # Wait for 1 minute before retrying
    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"❌ [cell_002] Unexpected error: {e}")
        time.sleep(60)  # Wait for 1 minute before retrying