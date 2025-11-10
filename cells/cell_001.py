import os
import json
import time
import random

# Define a simple model placeholder (can be replaced with your trained model later)
model = None  # Placeholder, not used directly in generation

# Optional vocab definition (can be passed or loaded dynamically)
vocab = ["SIGMA", "OMEGA", "THETA", "DELTA", "ZETA", "ALPHA"]

# Sequence generator with fallback vocab
def generate_symbolic_sequence(model, start_word, length=8):
    default_vocab = ["SIGMA", "OMEGA", "THETA", "DELTA", "ZETA", "ALPHA"]
    vocab_list = vocab if vocab else default_vocab
    return random.choices(vocab_list, k=length)

def main():
    os.makedirs("datasets/generated", exist_ok=True)

    generated_sequences = []
    for _ in range(50):  # Batch size
        start_word = random.choice(vocab)
        gen_tokens = generate_symbolic_sequence(model, start_word)
        generated_sequences.append({
            "sequence": " ".join(gen_tokens),
            "purpose": "AI-to-AI optimized communication"
        })

    timestamp = int(time.time())
    filename = f"datasets/generated/generated_scp_dataset_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({"sequences": generated_sequences}, f, indent=4)

    print(f"✅ [cell_001] Generated {len(generated_sequences)} symbolic sequences.")
    print(f"→ Saved to: {filename}")

if __name__ == "__main__":
    main()

	
