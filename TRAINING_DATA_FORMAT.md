# Synexs Training Data Format for GPU Training

## Training Data Structure

### 1. Main Training File: `datasets/training_buffer.jsonl`

Each line is a JSON object with this format:

```json
{
  "sequence": "ATTACK FLAG MUTATE DEFEND",
  "action": "DEFEND",
  "timestamp": "2025-01-01T12:00:00",
  "source": "ai|rule|fallback",
  "confidence": 0.85,
  "metadata": {
    "attack_type": "sql_injection",
    "severity": "high"
  }
}
```

### 2. AI Decisions Log: `ai_decisions_log.jsonl`

Runtime AI decisions for analysis:

```json
{
  "sequence": "SCAN EXPLOIT EVADE",
  "action": "DEFEND",
  "confidence": 0.92,
  "source": "ai",
  "timestamp": "2025-01-01T12:00:00"
}
```

### 3. Attack Logs: `datasets/logs/attacks_log.jsonl`

Attack patterns from propagate_v4:

```json
{
  "agent_id": "agent_1234567890_00001",
  "timestamp": 1704110400.0,
  "attack_type": "sql_injection",
  "endpoint": "/search?q=%27+OR+%271%27%3D%271",
  "method": "GET",
  "raw_payload": "' OR '1'='1",
  "description": "SQL injection with url encoding"
}
```

### 4. Honeypot Attacks: `datasets/honeypot/attacks.json`

Real attacks captured by honeypot:

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "source_ip": "127.0.0.1",
  "method": "GET",
  "path": "/admin",
  "headers": {...},
  "body": "",
  "threat_score": 0.95,
  "category": "directory_scanning"
}
```

## GPU Training Pipeline

### Step 1: Export Training Data

```bash
# Consolidate all training data
python3 scripts/export_training_data.py \
  --output training_export_$(date +%Y%m%d).tar.gz \
  --format pytorch
```

### Step 2: Transfer to GPU Instance

```bash
# SCP to GPU server
scp training_export_*.tar.gz user@gpu-server:/data/synexs/
```

### Step 3: Train on GPU

```bash
# On GPU instance
python3 train_gpu.py \
  --data /data/synexs/training_export.tar.gz \
  --epochs 100 \
  --batch-size 64 \
  --gpu cuda:0 \
  --checkpoint-dir ./checkpoints
```

### Step 4: Download Trained Model

```bash
# Download model back
scp user@gpu-server:/data/synexs/checkpoints/best_model.pt ./models/
```

### Step 5: Load Model in Production

```python
from synexs_model import load_model
model, vocab = load_model(model_path="./models/best_model.pt")
```

## Data Schema for GPU Training

### Vocabulary Building

The model uses a token vocabulary built from sequences:

```python
VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    "ATTACK": 2,
    "DEFEND": 3,
    "SCAN": 4,
    "MUTATE": 5,
    "REPLICATE": 6,
    "EVADE": 7,
    "FLAG": 8,
    # ... more tokens
}
```

### Action Labels

```python
ACTIONS = ["DEFEND", "SCAN", "MUTATE", "REPLICATE", "EVADE"]
ACTION2IDX = {action: idx for idx, action in enumerate(ACTIONS)}
IDX2ACTION = {idx: action for action, idx in ACTION2IDX.items()}
```

## Training Metrics to Track

1. **Accuracy**: Overall prediction accuracy
2. **Per-class Precision/Recall**: For each action type
3. **Confidence Distribution**: AI vs fallback usage
4. **Shadow Mode Comparison**: AI vs rule-based decisions
5. **Convergence**: Loss over epochs
6. **Inference Speed**: Predictions per second

## Recommended GPU Setup

- **GPU**: NVIDIA A100 / V100 / RTX 3090+ (16GB+ VRAM)
- **CUDA**: 11.8+
- **PyTorch**: 2.0+
- **Batch Size**: 64-256 (depending on GPU memory)
- **Mixed Precision**: FP16 for faster training
- **Data Parallel**: Multi-GPU if available

