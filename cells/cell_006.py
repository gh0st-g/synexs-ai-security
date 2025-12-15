#!/usr/bin/env python3
"""
cell_006.py - Synexs Symbolic Sequence Classifier
This script loads refined symbolic sequences from JSON files, classifies them using a pre-trained model,
and saves the predicted actions to a decisions JSON file. Optimized for efficiency with batch processing,
error handling, and minimal memory usage.
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components from synexs_model
from synexs_model import SynexsCoreModel, IDX2ACTION, load_model, load_vocab, SynexsDataset, collate_fn

# Configuration Constants
MODEL_PATH = "synexs_core_model.pth"
REFINED_DIR = "datasets/refined"
DECISIONS_PATH = "datasets/decisions/decisions.json"

# Ensure output directories exist
os.makedirs(os.path.dirname(DECISIONS_PATH), exist_ok=True)

# Load vocab and model
vocab = load_vocab()
vocab_size = max(vocab.values()) + 1  # Use max index + 1, not len(vocab)
num_actions = len([k for k in vocab.keys() if not k.startswith('<')])
model = SynexsCoreModel(vocab_size, output_dim=num_actions)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Use CPU for efficiency on VPS
model.eval()

# Classification Function (batch processing for optimization)
def classify_sequences(sequences):
    """
    Classifies a list of symbolic sequences using the pre-trained model.

    Args:
        sequences (list): List of sequence strings or dicts with 'sequence' keys.

    Returns:
        list: Predicted actions for each sequence.
    """
    if not sequences:
        return []

    dataset = SynexsDataset(sequences, vocab)
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, num_workers=0)  # num_workers=0 to avoid overhead on low-RAM VPS
    predictions = []

    with torch.no_grad():  # Disable gradients for memory optimization
        for x_batch, _ in loader:
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).tolist()
            predictions.extend([IDX2ACTION[p] for p in preds])

    return predictions

# Main Function
def main():
    """
    Main function to load refined sequences, classify them, and save decisions.
    Handles various JSON structures robustly with error handling.
    """
    sequences = []
    for fname in os.listdir(REFINED_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(REFINED_DIR, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Handle different JSON structures efficiently
            seq_data = data.get("sequences", []) if isinstance(data, dict) else data if isinstance(data, list) else []
            for entry in seq_data:
                if isinstance(entry, dict):
                    # Try both "sequence" and "refined" keys
                    seq = entry.get("sequence") or entry.get("refined") or ""
                elif isinstance(entry, str):
                    seq = entry
                if seq:
                    sequences.append(seq)
            print(f"✅ [cell_006] Loaded {len(seq_data)} items from {fname}")
        except json.JSONDecodeError as e:
            print(f"❌ [cell_006] JSON decode error in {fname}: {e}")
        except Exception as e:
            print(f"❌ [cell_006] Error processing {fname}: {e}")

    if sequences:
        decisions = classify_sequences(sequences)
        # Pair sequences with their predicted actions
        decision_pairs = []
        for seq, action in zip(sequences, decisions):
            decision_pairs.append({
                "sequence": seq,
                "action": action
            })

        try:
            with open(DECISIONS_PATH, "w") as f:
                json.dump({"decisions": decision_pairs}, f, indent=4)
            print(f"✅ [cell_006] Classified {len(sequences)} sequences and saved {len(decision_pairs)} decisions.")

            # Print action distribution for monitoring
            action_counts = {}
            for action in decisions:
                action_counts[action] = action_counts.get(action, 0) + 1
            print(f"   Action distribution: {action_counts}")
        except Exception as e:
            print(f"❌ [cell_006] Error saving decisions: {e}")
    else:
        print("⚠️ [cell_006] No sequences to classify.")

if __name__ == "__main__":
    main()
