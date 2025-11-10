#!/usr/bin/env python3
"""
Synexs Unified Model Module
Consolidates all ML model code for symbolic sequence classification
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset
import logging
import os

# ==================== Configuration ====================
MODEL_PATH = "synexs_core_model.pth"
VOCAB_PATH = "vocab.json"

# Action mappings
ACTIONS = ["discard", "refine", "replicate", "mutate", "flag"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for a, i in ACTION2IDX.items()}

# ==================== Model Architecture ====================
class SynexsCoreModel(nn.Module):
    """
    Neural network for symbolic sequence classification
    Architecture: Embedding → FC1 → ReLU → Dropout → FC2
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, output_dim=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass"""
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        h = self.fc1(pooled)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return out

# ==================== Helper Functions ====================
def collate_fn(batch):
    """Collate function for DataLoader to handle variable length sequences"""
    sequences, labels = zip(*batch)
    # Pad sequences to same length
    max_len = max(len(seq) for seq in sequences)
    padded = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in sequences]
    return torch.stack(padded), torch.tensor(labels)

# ==================== Dataset Class ====================
class SynexsDataset(Dataset):
    """Dataset for loading symbolic sequences"""
    def __init__(self, sequences: List[str], vocab: Optional[Dict[str, int]] = None):
        self.sequences = sequences
        # If vocab not provided, will be set later or use default
        self.vocab = vocab if vocab is not None else {}
        self.unk_idx = self.vocab.get("<UNK>", 0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if isinstance(seq, dict):
            seq = seq.get("sequence", "")
        tokens = seq.split()
        token_ids = [self.vocab.get(tok, self.unk_idx) for tok in tokens]
        return torch.tensor(token_ids, dtype=torch.long), 0  # Return dummy label for compatibility

# ==================== Model Loading ====================
def load_vocab(vocab_path: str = VOCAB_PATH) -> Dict[str, int]:
    """Load vocabulary from JSON file"""
    try:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab
    except FileNotFoundError:
        logging.error(f"Vocabulary file not found: {vocab_path}")
        raise
    except Exception as e:
        logging.error(f"Failed to load vocabulary: {e}")
        raise

def load_model(model_path: str = MODEL_PATH, vocab: Optional[Dict] = None) -> tuple:
    """Load trained model and vocabulary"""
    if vocab is None:
        vocab = load_vocab()

    vocab_size = len(vocab)
    model = SynexsCoreModel(vocab_size)

    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model, vocab
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

# ==================== Inference Functions ====================
def predict_action(sequence: str, model: nn.Module, vocab: Dict[str, int]) -> str:
    """Predict action for a symbolic sequence"""
    try:
        tokens = sequence.split()
        unk_idx = vocab.get("<UNK>", len(vocab) - 1)
        token_ids = [vocab.get(tok, unk_idx) for tok in tokens]
        x = torch.tensor([token_ids], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        return IDX2ACTION[pred_idx]
    except Exception as e:
        logging.error(f"Error in predict_action: {e}")
        return "discard"

# ==================== Main ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("🧠 Synexs Model Module - Testing")
    try:
        model, vocab = load_model()
        logging.info(f"✅ Model loaded (vocab size: {len(vocab)})")
        test_seq = "SIGMA OMEGA THETA"
        action = predict_action(test_seq, model, vocab)
        logging.info(f"Test: '{test_seq}' → {action}")
    except Exception as e:
        logging.error(f"❌ Error: {e}")