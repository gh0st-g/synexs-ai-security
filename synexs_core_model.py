import torch
import torch.nn as nn
import logging

# ===============================
# Synexs Core Neural Architecture (Symbolic Engine)
# ===============================

class SynexsCoreModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        embedded_mean = embedded.mean(dim=1)
        h = self.relu(self.fc1(embedded_mean))
        output = self.fc2(h)
        return output

# ===============================
# Example usage (Testing Mode)
# ===============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    try:
        vocab_size = 100
        model = SynexsCoreModel(vocab_size)
        device = next(iter(model.parameters())).device
        dummy_input = torch.randint(0, vocab_size, (2, 10), device=device)
        output = model(dummy_input)
        logging.info("Output probabilities: %s", output)
        logging.info("✅ Synexs Core Model initialized.")
    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        raise