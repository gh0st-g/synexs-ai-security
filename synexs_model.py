#!/usr/bin/env python3
"""
Synexs Unified Model Module - V3 Protocol Support
Consolidates all ML model code for symbolic sequence classification
Supports V3 Binary Protocol (32 actions)
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
VOCAB_PATH = "vocab_v3_binary.json"  # V3 Protocol vocabulary

# ==================== Vocabulary Loading ====================
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

# V3 Protocol: 32 Actions (loaded dynamically from vocab)
def _load_action_mappings():
    """Load V3 protocol action mappings from vocabulary"""
    try:
        vocab = load_vocab()
        # Extract action names (non-special tokens)
        actions = [k for k in vocab.keys() if not k.startswith('<')]
        action2idx = {a: vocab[a] for a in actions}
        idx2action = {v: k for k, v in action2idx.items()}
        return actions, action2idx, idx2action
    except Exception as e:
        logging.warning(f"Failed to load V3 actions, using defaults: {e}")
        # Fallback to minimal set
        default_actions = ["SCAN", "ATTACK", "REPLICATE", "MUTATE", "EVADE"]
        default_action2idx = {a: i for i, a in enumerate(default_actions)}
        default_idx2action = {i: a for a, i in default_action2idx.items()}
        return default_actions, default_action2idx, default_idx2action

ACTIONS, ACTION2IDX, IDX2ACTION = _load_action_mappings()

# ==================== Model Architecture ====================
class SynexsCoreModel(nn.Module):
    """
    Neural network for symbolic sequence classification - V3 Protocol
    Architecture: Embedding → FC1 → ReLU → Dropout → FC2
    Supports 32 action outputs for V3 binary protocol
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, output_dim=32):
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
def load_model(model_path: str = MODEL_PATH, vocab: Optional[Dict] = None) -> tuple:
    """Load trained model and vocabulary - V3 Protocol Support"""
    if vocab is None:
        vocab = load_vocab()

    vocab_size = max(vocab.values()) + 1  # Dynamic vocab size
    output_dim = len([k for k in vocab.keys() if not k.startswith('<')])  # Number of actions

    model = SynexsCoreModel(vocab_size, output_dim=output_dim)

    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        logging.info(f"Loaded V3 model: vocab_size={vocab_size}, output_dim={output_dim}")
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