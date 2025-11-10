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
MODEL_SAVE_PATH = "core_model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

ACTIONS = ["discard", "refine", "replicate", "mutate", "flag"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for a, i in ACTION2IDX.items()}

# 20-token vocab
vocab = {
    "[SIGNAL]": 0,
    "+LANG@SYNEXS": 1,
    "[ROLE]": 2,
    "AI": 3,
    "[ACTION]": 4,
    "VERIFY": 5,
    "HASH:": 6,
    "CAPSULE_08_FINAL": 7,
    "<EOS>": 8,
    "<UNK>": 9,
    "+Ψ": 10,
    "@CORE": 11,
    "∆SIG": 12,
    "[RECURSE]": 13,
    "NODE": 14,
    "[TRACE]": 15,
    "REPLICATE": 16,
    "CELL": 17,
    "UPLINK": 18,
    "[TAG]": 19
}
vocab_size = len(vocab)

class SynexsCoreModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, output_dim=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_mean = embedded.mean(dim=1)
        h = self.fc1(embedded_mean)
        h = self.relu(h)
        logits = self.fc2(h)
        return logits

class SynexsDataset(Dataset):
    def __init__(self, data):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = item["sequence"].split()
        token_ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        x = torch.tensor(token_ids, dtype=torch.long)
        y = ACTION2IDX[item["action"]]
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = [torch.cat([x, torch.full((max_len - len(x),), vocab["<EOS>"])]) for x in xs]
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

    model = SynexsCoreModel(vocab_size)
    print("[CORE AI] Training model with 20-token vocab...")
    model = train_model(model, loader, epochs=EPOCHS, lr=LR)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[CORE AI] Saved model to {MODEL_SAVE_PATH}")
