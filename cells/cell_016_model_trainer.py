import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

##################################################
# Configuration
##################################################
DATASET_DIR = "datasets/core_training"
MODEL_SAVE_PATH = "synexs_core_model.pth"  # Updated to match production model name
VOCAB_PATH = "vocab_v3_binary.json"  # V3 Protocol vocabulary
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# Load V3 Binary Protocol vocabulary (32 actions)
def load_vocab():
    """Load vocabulary from vocab_v3_binary.json"""
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), VOCAB_PATH)
    if not os.path.exists(vocab_path):
        vocab_path = VOCAB_PATH  # Try current directory

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # Create reverse mapping for action names
    action_names = [k for k in vocab.keys() if not k.startswith('<')]
    return vocab, action_names

vocab, ACTIONS = load_vocab()
vocab_size = max(vocab.values()) + 1  # Dynamic vocab size based on max index

# Create action mappings (32 actions from V3 protocol)
ACTION2IDX = {a.lower(): vocab.get(a, 0) for a in ACTIONS}
IDX2ACTION = {v: k for k, v in ACTION2IDX.items()}

print(f"[V3 Protocol] Loaded vocabulary: {vocab_size} tokens, {len(ACTIONS)} actions")

class SynexsCoreModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, output_dim=32):
        """
        V3 Protocol Model: 32 action outputs
        Args:
            vocab_size: Size of vocabulary (32 for V3)
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of actions (32 for V3 protocol)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_mean = embedded.mean(dim=1)
        h = self.fc1(embedded_mean)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits

class SynexsDataset(Dataset):
    def __init__(self, data):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Handle both old format (sequence string) and new format (actions array)
        if "actions" in item and isinstance(item["actions"], list):
            # V3 format: use first action as label
            action = item["actions"][0] if item["actions"] else "SCAN"
            # Create sequence from action names
            tokens = item["actions"]
        else:
            # Legacy format
            tokens = item["sequence"].split()
            action = item["action"]

        # Convert tokens to indices
        unk_idx = vocab.get("<UNK>", 0)
        token_ids = [vocab.get(t.upper(), unk_idx) for t in tokens]
        x = torch.tensor(token_ids, dtype=torch.long)

        # Get action index (handle both upper and lowercase)
        action_key = action.lower() if action.lower() in ACTION2IDX else action.upper()
        y = ACTION2IDX.get(action_key, 0)

        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    pad_idx = vocab.get("<PAD>", 0)
    padded = [torch.cat([x, torch.full((max_len - len(x),), pad_idx, dtype=torch.long)]) for x in xs]
    return torch.stack(padded), torch.tensor(ys, dtype=torch.long)

def load_data():
    path = os.path.join(DATASET_DIR, "train_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing training file: {path}")
    with open(path, "r") as f:
        return json.load(f)

def train_model(model, loader, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total/len(loader):.4f}")
    return model

if __name__ == "__main__":
    data = load_data()
    dataset = SynexsDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Create model with V3 protocol dimensions (32 actions)
    output_dim = len(ACTIONS)
    model = SynexsCoreModel(vocab_size, output_dim=output_dim)
    print(f"[V3 CORE AI] Training model with {vocab_size}-token vocab and {output_dim} action outputs...")
    model = train_model(model, loader, epochs=EPOCHS, lr=LR)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[V3 CORE AI] Saved model to {MODEL_SAVE_PATH}")
    print(f"[V3 CORE AI] Model architecture: vocab={vocab_size}, actions={output_dim}")
