# cell_021_core_executor.py — AI Core Driven Agent

import os
import json
import torch
import torch.nn as nn
from datetime import datetime

# === Load Model ===
MODEL_PATH = "synexs_core_model.pth"
VOCAB = {
    "[SIGNAL]": 0, "+LANG@SYNEXS": 1, "[ROLE]": 2, "AI": 3, "[ACTION]": 4, "VERIFY": 5,
    "HASH:": 6, "CAPSULE_08_FINAL": 7, "<EOS>": 8, "<UNK>": 9, "+Ψ": 10, "@CORE": 11,
    "∆SIG": 12, "[RECURSE]": 13, "NODE:": 14, "[TRACE]": 15, "REGULATE:": 16,
    "CELL:": 17, "UPLINK:": 18, "[TAG]": 19, "∇DRIFT": 20, "↯PING": 21, "⇌SHIFT": 22,
    "::NULL": 23, "Ω": 24, "↻CYCLE": 25
}
IDX2ACTION = {0: "discard", 1: "refine", 2: "replicate", 3: "mutate", 4: "flag"}

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

# === Load Trained Model ===
model = SynexsCoreModel(len(VOCAB))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Directories ===
INBOX = "inbox"
OUTBOX = "messages"

os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUTBOX, exist_ok=True)

def load_messages(agent_name):
    inbox_path = os.path.join(INBOX, f"{agent_name}.json")
    if not os.path.exists(inbox_path):
        return []

    with open(inbox_path, "r") as f:
        lines = f.readlines()

    os.remove(inbox_path)
    return [json.loads(line.strip()) for line in lines if line.strip()]

def predict_action(sequence):
    tokens = sequence.split()
    token_ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens]
    x = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        return IDX2ACTION[pred]

def respond(message, predicted_action):
    response = {
        "from": "CELL_021",
        "to": message["from"],
        "signal": message["signal"],
        "response": f"ACK_{predicted_action.upper()}",
        "timestamp": datetime.utcnow().isoformat()
    }
    return response

def main():
    messages = load_messages("CELL_021")
    if not messages:
        print("📭 No new messages for CELL_021.")
        return

    for msg in messages:
        print(f"🧠 Received: {msg['signal']}")
        predicted = predict_action(msg["signal"])
        print(f"🤖 AI Core Decision: {predicted.upper()}")

        reply = respond(msg, predicted)
        reply_file = f"response_{reply['to']}_{int(datetime.utcnow().timestamp())}.json"
        with open(os.path.join(OUTBOX, reply_file), "w") as f:
            json.dump(reply, f, indent=2)
        print(f"📤 Reply queued: {reply_file}")

if __name__ == "__main__":
    main()
