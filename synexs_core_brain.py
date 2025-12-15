import torch
import torch.nn as nn
import json
from datetime import datetime
import os
from pathlib import Path
import logging

# ========== Load Vocab ==========
try:
    with open("vocab.json", "r") as f:
        vocab = json.load(f)
except FileNotFoundError:
    logging.error("Error: vocab.json file not found.")
    exit(1)
vocab_size = len(vocab)

# ========== Action Map ==========
ACTIONS = ["discard", "refine", "replicate", "mutate", "flag"]
IDX2ACTION = {i: a for i, a in enumerate(ACTIONS)}

# ========== Synexs Core Model ==========
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
        out = self.fc2(h)
        return out

# ========== Load Model ==========
try:
    model = SynexsCoreModel(vocab_size)
    model.load_state_dict(torch.load("synexs_core_model.pth", map_location=torch.device("cpu")))
    model.eval()
except FileNotFoundError:
    logging.error("Error: synexs_core_model.pth file not found.")
    exit(1)

# ========== Inference ==========
def tokenize_input(text):
    tokens = text.strip().split()
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    return torch.tensor([ids], dtype=torch.long)

def generate_agent_plan(symbolic_input):
    try:
        x = tokenize_input(symbolic_input)
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
            action = IDX2ACTION[pred]

        task_map = {
            "discard": "Archive input and halt processing.",
            "refine": "Analyze and optimize input sequence.",
            "replicate": "Create an inbox scanner that summarizes unread messages.",
            "mutate": "Generate a reporting bot for weekly email summaries.",
            "flag": "Deploy a Gmail agent that flags VIPs and removes spam."
        }
        task = task_map.get(action, "Build an auto-responder for common client inquiries.")

        log_entry = {
            "input": symbolic_input,
            "action": action,
            "task": task,
            "timestamp": datetime.utcnow().isoformat()
        }
        log_dir = "logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, "brain_log.json")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return task
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return "Error occurred during processing."

# ========== CLI Test Loop ==========
if __name__ == "__main__":
    logging.basicConfig(filename="synexs_core_brain.log", level=logging.ERROR)
    print("🧠 Synexs Core Brain Ready.")
    while True:
        try:
            inp = input(">> Symbolic Input: ")
            if inp.lower() in ["exit", "quit"]:
                break
            result = generate_agent_plan(inp)
            print(f"🤖 Output: {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f"❌ Error: {e}")
            print(f"❌ Error: {e}")