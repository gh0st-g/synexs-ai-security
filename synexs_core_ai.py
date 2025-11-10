import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ====================
# Paths & Hyperparams
# ====================
DATASET_PATH = "datasets/core_training/train_data.json"
MODEL_SAVE_PATH = "synexs_core_model.pth"
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

# ====================
# Action Map
# ====================
ACTIONS = ["discard", "refine", "replicate", "mutate", "flag"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for a, i in ACTION2IDX.items()}

# ====================
# Load Data & Vocab
# ====================
try:
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
except (FileNotFoundError, IOError, OSError) as e:
    print(f"Error: Could not load data from {DATASET_PATH}. {e}")
    exit(1)

# Build vocab dynamically
tokens = set(token for entry in data for token in entry["sequence"].split())
vocab = {token: i for i, token in enumerate(sorted(tokens))}
vocab["<EOS>"] = len(vocab)
vocab["<UNK>"] = len(vocab)
vocab_size = len(vocab)

# ====================
# Dataset Definition
# ====================
class SynexsDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        sequence = entry.get("sequence", "")
        action = entry.get("action", "discard")
        token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in sequence.split()]
        x = torch.tensor(token_ids, dtype=torch.long)
        y = ACTION2IDX[action]
        return x, y

# ====================
# Collate Function
# ====================
def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_xs = [torch.cat([x, torch.full((max_len - len(x),), vocab["<EOS>"], dtype=torch.long)]) for x in xs]
    return torch.stack(padded_xs), torch.tensor(ys, dtype=torch.long)

# ====================
# Feedforward Model
# ====================
class SynexsCoreModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        pooled = emb.mean(dim=1)
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)

# ====================
# Training
# ====================
dataset = SynexsDataset(data)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SynexsCoreModel(vocab_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    acc = 100 * correct / total
    print(f"📚 Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f} — Accuracy: {acc:.2f}%")

try:
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model saved as {MODEL_SAVE_PATH}")
except (IOError, OSError) as e:
    print(f"\nError: Could not save model to {MODEL_SAVE_PATH}. {e}")