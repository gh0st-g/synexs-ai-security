# ==============================
# CELL_MUTATOR — Symbolic Evolver
# ==============================

import os
import json
import random
from datetime import datetime

INBOX_FILE = "inbox/CELL_MUTATOR.json"
MEMORY_FILE = "memory_log.json"
RESPONSE_DIR = "inbox"
VOCAB = [
    "[SIGNAL]", "+LANG@SYNEXS", "[ROLE]", "AI", "[ACTION]",
    "VERIFY", "HASH:", "CAPSULE_08_FINAL", "<EOS>", "<UNK>",
    "+Ψ", "@CORE", "∆SIG", "[RECURSE]", "NODE:",
    "[TRACE]", "REGULATE:", "CELL:", "UPLINK:", "[TAG]"
]

def load_inbox():
    if not os.path.exists(INBOX_FILE):
        return []
    with open(INBOX_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def mutate_sequence(signal):
    tokens = signal.split()
    new_tokens = []
    for t in tokens:
        if random.random() < 0.2:
            new_tokens.append(random.choice(VOCAB))
        else:
            new_tokens.append(t)
    if random.random() < 0.3:
        new_tokens.append(random.choice(VOCAB))
    return " ".join(new_tokens)

def respond_to(sender, original_signal, mutated_signal):
    response = {
        "from": "CELL_MUTATOR",
        "to": sender,
        "signal": mutated_signal,
        "response": f"Mutated from: {original_signal}",
        "timestamp": datetime.utcnow().isoformat()
    }
    inbox_file = os.path.join(RESPONSE_DIR, f"{sender}.json")
    with open(inbox_file, "a") as f:
        f.write(json.dumps(response) + "\n")
    print(f"🧬 Sent mutation to {sender}: {mutated_signal}")

def process():
    messages = load_inbox()
    memory = load_memory()
    if not messages:
        print("📭 No messages for CELL_MUTATOR.")
        return

    for msg in messages:
        signal = msg.get("signal", "")
        sender = msg.get("from", "UNKNOWN")
        mutated = mutate_sequence(signal)
        respond_to(sender, signal, mutated)

    open(INBOX_FILE, "w").close()

if __name__ == "__main__":
    process()
