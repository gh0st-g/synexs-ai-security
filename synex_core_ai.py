from flask import Flask, request, jsonify
import json
import torch
import torch.nn as nn
from os.path import isfile
import logging
from datetime import datetime
import os

app = Flask(__name__)
logging.basicConfig(filename='synexs_log.out', level=logging.INFO)

vocab = {
    "[SIGNAL]": 0, "+LANG@SYNEXS": 1, "[ROLE]": 2, "AI": 3, "[ACTION]": 4,
    "VERIFY": 5, "HASH:": 6, "CAPSULE_08_FINAL": 7, "<EOS>": 8, "<UNK>": 9,
    "+Ψ": 10, "@CORE": 11, "∆SIG": 12, "[RECURSE]": 13, "NODE": 14, "[TRACE]": 15,
    "REPLICATE": 16, "CELL": 17, "UPLINK": 18, "[TAG]": 19
}
vocab_size = len(vocab)

ACTIONS = ["discard", "refine", "replicate", "mutate", "flag"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for a, i in ACTION2IDX.items()}

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

model_path = "core_model.pth"
if os.path.isfile(model_path):
    model = SynexsCoreModel(vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    logging.error("Error: Model file not found.")
    raise FileNotFoundError("Model file not found.")

def predict_action(sequence_str):
    tokens = sequence_str.split()
    token_ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    x = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        prediction = torch.argmax(logits, dim=1).item()
    return IDX2ACTION[prediction]

@app.route('/')
def home():
    return '🧠 Synexs Core AI Dashboard is Live'

@app.route('/logs')
def logs():
    log_file = 'synexs_log.out'
    if isfile(log_file):
        with open(log_file, 'r') as f:
            content = f.readlines()[-100:]
        return '<pre>' + ''.join(content) + '</pre>'
    else:
        logging.error("Error: No log data found.")
        return jsonify({"error": "No log data found."}), 404

@app.route('/blockchain')
def blockchain():
    blockchain_file = 'blockchain_log.json'
    if isfile(blockchain_file):
        with open(blockchain_file) as f:
            data = json.load(f)
        return jsonify(data)
    else:
        logging.error("Error: No blockchain log found.")
        return jsonify({"error": "No blockchain log found."}), 404

@app.route('/predict')
def predict():
    sequence = request.args.get('sequence')
    if not sequence:
        logging.error("Error: Missing 'sequence' parameter")
        return jsonify({"error": "Missing 'sequence' parameter"}), 400
    try:
        action = predict_action(sequence)
        return jsonify({
            "sequence": sequence,
            "predicted_action": action
        })
    except Exception as e:
        logging.error(f"Error: {str(e)} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)