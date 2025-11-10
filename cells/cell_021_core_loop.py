import os
import json
import time
import torch
from datetime import datetime
from synexs_core_model import SynexsCoreModel

# Load vocab
with open("vocab.json") as f:
    vocab = json.load(f)

# Load trained model
model = SynexsCoreModel(vocab_size=len(vocab))
model.load_state_dict(torch.load("synexs_core_model.pth", map_location=torch.device("cpu")))
model.eval()

INBOX_DIR = "inbox"
RESP_DIR = "responses"
os.makedirs(RESP_DIR, exist_ok=True)

def encode_sequence(seq, vocab):
    tokens = seq.split()
    return torch.tensor([[vocab.get(t, vocab["<UNK>"]) for t in tokens]])

def interpret_sequence(seq):
    x = encode_sequence(seq, vocab)
    with torch.no_grad():
        logits, _ = model(x)
        prediction = torch.argmax(logits, dim=1).item()
    labels = ["replicate", "mutate", "refine", "discard", "flag"]
    return labels[prediction]

def process_messages():
    inbox_file = os.path.join(INBOX_DIR, "CELL_021.json")
    if not os.path.exists(inbox_file):
        return

    with open(inbox_file, "r") as f:
        lines = f.readlines()

    if not lines:
        return

    with open(inbox_file, "w") as f:
        f.truncate(0)  # Clear inbox

    for line in lines:
        msg = json.loads(line)
        signal = msg.get("signal", "")
        sender = msg.get("from", "UNKNOWN")
        decision = interpret_sequence(signal)

        # Create response
        response = {
            "from": "CELL_021",
            "to": sender,
            "signal": signal,
            "response": decision.upper(),
            "timestamp": datetime.utcnow().isoformat()
        }

        # Save reply
        inbox_reply = os.path.join(INBOX_DIR, f"{sender}.json")
        with open(inbox_reply, "a") as f:
            f.write(json.dumps(response) + "\n")

        print(f"🧠 Received: {signal}")
        print(f"🤖 AI Core Decision: {decision.upper()}")
        print(f"📤 Replied to {sender} with: {decision.upper()}")

        # Log the interaction
        log_entry = {
            "input": signal,
            "decision": decision,
            "timestamp": datetime.utcnow().isoformat()
        }
        log_path = os.path.join(RESP_DIR, f"log_{int(time.time())}.json")
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)

# 🔁 Inference loop
if __name__ == "__main__":
    print("🧠 Synexs Core Loop started...")
    while True:
        process_messages()
        time.sleep(3)
