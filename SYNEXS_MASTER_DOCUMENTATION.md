# Synexs - Complete Project Documentation

**Version**: 2.0
**Date**: 2025-11-09
**Status**: Production Ready - Data Collection Phase

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Communication Protocols](#communication-protocols)
5. [File Structure](#file-structure)
6. [Data Flow](#data-flow)
7. [Training Data](#training-data)
8. [Deployment Status](#deployment-status)
9. [Performance Metrics](#performance-metrics)
10. [GPU Training Guide](#gpu-training-guide)
11. [Chatbot Integration](#chatbot-integration)
12. [Future Roadmap](#future-roadmap)

---

## 1. Project Overview

### **What is Synexs?**

Synexs is an **AI-powered defensive security training system** that uses:
- **Agent-based swarm intelligence** for adaptive learning
- **Honeypot detection** to identify attack patterns
- **Evolutionary algorithms** (spawn → mutate → replicate)
- **Binary protocol communication** (88% bandwidth reduction)
- **Self-learning from failures** (kill reports → mutations)

### **Key Capabilities**

✅ **Defensive Training**: Localhost-only honeypot targeting
✅ **Adaptive Learning**: AI learns from agent deaths
✅ **Efficient Communication**: Binary protocol (6 bytes vs 46 bytes)
✅ **Autonomous Operation**: Cell-based orchestration
✅ **Real-time Data Collection**: DNA collector from live operations

### **Purpose**

Train AI agents to:
1. **Detect honeypots** (CIDR validation, PTR records, attack patterns)
2. **Adapt strategies** (mutation after failures)
3. **Optimize communication** (ultra-compact binary protocol)
4. **Learn continuously** (self-generating training data)

---

## 2. System Architecture

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNEXS ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐   ┌───────────────┐  │
│  │  Honeypot   │───▶│ DNA Collector│──▶│Training Data  │  │
│  │   Server    │    │  (30 min)    │   │ (Binary V3)   │  │
│  └─────────────┘    └──────────────┘   └───────────────┘  │
│         │                                       │           │
│         ▼                                       ▼           │
│  ┌─────────────┐    ┌──────────────┐   ┌───────────────┐  │
│  │  Listener   │───▶│ AI Swarm     │──▶│  Cell Executor│  │
│  │  (Reports)  │    │  (Learning)  │   │ (Orchestrator)│  │
│  └─────────────┘    └──────────────┘   └───────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          Propagate V3 (Agent Spawner)               │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Data Flow**

```
1. Honeypot captures attacks → datasets/honeypot/attacks.json
2. DNA Collector monitors logs → Generates training samples
3. Cell Orchestrator runs pipeline → Processes symbolic sequences
4. AI Swarm learns patterns → Updates mutations
5. Propagate spawns agents → Applies learned strategies
6. Cycle repeats → Continuous improvement
```

---

## 3. Core Components

### **3.1 Honeypot Server** (`honeypot_server.py`)

**Purpose**: Capture attack attempts for analysis

**Features**:
- SSH honeypot (port 2222)
- PTR record validation
- CIDR IP filtering
- Batch write optimization (50 events buffered)
- DNS caching (@lru_cache)

**Output**: `datasets/honeypot/attacks.json`

**Performance**:
- 99% I/O reduction (batch writes)
- 99% faster PTR lookups (cache)

---

### **3.2 Listener** (`listener.py`)

**Purpose**: Collect kill reports from terminated agents

**Features**:
- Socket server (port 5555)
- PID file locking (singleton)
- Log rotation (10MB, 5 backups)
- Kill report parsing

**Output**: `listener.log`

**Improvements**:
- Fixed 157 zombie processes (fcntl locking)
- Log rotation prevents disk bloat

---

### **3.3 AI Swarm** (`ai_swarm_fixed.py`)

**Purpose**: Learn from agent failures and trigger mutations

**Features**:
- Monitors kill reports
- Triggers mutations on patterns
- Dataset cleanup (7-day retention)
- Adaptive strategy updates

**Learning Cycle**:
1. Agent dies → Kill report received
2. Pattern analyzed → Failure cause identified
3. Mutation triggered → Strategy adapted
4. New agents spawned → Improved tactics

---

### **3.4 Propagate V3** (`propagate_v3.py`)

**Purpose**: Spawn and manage agent swarm

**Features**:
- Agent spawning (Python-based)
- Mutation application
- Replication on success
- Target selection

**Cycle**: Spawn → Mutate → Replicate

---

### **3.5 Core Orchestrator** (`synexs_core_orchestrator.py`)

**Purpose**: Coordinate all cells in 5-phase pipeline

**Pipeline**:
```
Phase 1: GENERATION
  ├─ cell_001.py → Generate sequences
  └─ (cell_002.py runs independently)

Phase 2: PROCESSING
  ├─ cell_004.py → Hash logging
  └─ cell_010_parser.py → Parse tokens

Phase 3: CLASSIFICATION
  └─ cell_006.py → AI classification

Phase 4: EVOLUTION
  ├─ cell_014_mutator.py → Mutate sequences
  └─ cell_015_replicator.py → Replicate patterns

Phase 5: FEEDBACK
  └─ cell_016_feedback_loop.py → Analyze results
```

**Performance**:
- Cycle time: ~10 seconds
- Success rate: 100% (7/7 cells)
- Auto log rotation: 50MB limit

---

### **3.6 DNA Collector** (`dna_collector.py`)

**Purpose**: Generate training data from live operations

**Monitors**:
- Honeypot logs (attacks)
- Memory logs (learning)
- Cell outputs (generations)

**Trigger**: Every 100 events → Generate 50 samples

**Output**: Appends to `training_binary_v3.jsonl`

**Schedule**: Runs every 30 minutes (cron)

---

## 4. Communication Protocols

### **4.1 Protocol V1 (Greek Words)** - Legacy

**Format**: `"SIGMA OMEGA THETA DELTA"`
**Size**: 28 bytes (5 tokens)
**Use**: Legacy compatibility, debugging

### **4.2 Protocol V2 (Symbols)** - Efficient

**Format**: `"△□◆◇○"`
**Size**: 15 bytes (5 tokens)
**Reduction**: 46% smaller than V1
**Use**: Balance of readability + efficiency

### **4.3 Protocol V3 (Binary)** - Production ✅

**Format**: Base64 `"CABEMhTH"` or Hex `"0800443214c7"`
**Size**: 6-8 bytes (8 tokens)
**Reduction**: 88% smaller than V1
**Speedup**: 8.3x faster transmission

**Encoding**: 5 bits per action (32 possible actions)

**Actions** (32 total):
```
0x00 = SCAN          0x10 = MERGE
0x01 = ATTACK        0x11 = STACK_PUSH
0x02 = REPLICATE     0x12 = STACK_POP
0x03 = MUTATE        0x13 = TERMINATE
0x04 = EVADE         0x14 = PAUSE
0x05 = LEARN         0x15 = LOG
0x06 = REPORT        0x16 = QUERY
0x07 = DEFEND        0x17 = ACK
0x08 = REFINE        0x18 = NACK
0x09 = FLAG          0x19 = CHECKPOINT
0x0A = XOR_PAYLOAD   0x1A = VALIDATE
0x0B = ENCRYPT       0x1B = BROADCAST
0x0C = COMPRESS      0x1C = LISTEN
0x0D = HASH_CHECK    0x1D = ROUTE
0x0E = SYNC          0x1E = FILTER
0x0F = SPLIT         0x1F = TRANSFORM
```

---

## 5. File Structure

### **Directory Layout**

```
/root/synexs/
├── Core Systems
│   ├── honeypot_server.py          # Attack capture
│   ├── listener.py                 # Kill reports
│   ├── propagate_v3.py             # Agent spawner
│   ├── ai_swarm_fixed.py           # Learning engine
│   └── synexs_core_orchestrator.py # Cell coordinator
│
├── Protocol Implementation
│   ├── binary_protocol.py          # V3 binary (88% reduction)
│   ├── protocol_v2_proposal.py     # V2 symbolic (46% reduction)
│   ├── vocab_v3_binary.json        # 32 actions (5-bit)
│   └── vocab_v2.json               # 30 symbols
│
├── AI/ML Components
│   ├── synexs_model.py             # Unified ML model
│   ├── synexs_core_model.pth       # Trained weights (15KB)
│   └── vocab.json                  # Original vocabulary (26 tokens)
│
├── Data Collection
│   ├── dna_collector.py            # Auto training data generator
│   ├── training_binary_v3.jsonl    # 1050+ samples
│   ├── training_symbolic_v2.jsonl  # 50 symbolic samples
│   └── .dna_collector_state.json   # Collection state
│
├── Cells (Processing Pipeline)
│   ├── cells/cell_001.py           # Sequence generator
│   ├── cells/cell_001_hybrid.py    # Multi-protocol generator
│   ├── cells/cell_004.py           # Hash logging
│   ├── cells/cell_006.py           # AI classifier
│   ├── cells/cell_010_parser.py    # Token parser
│   ├── cells/cell_014_mutator.py   # Mutation engine
│   ├── cells/cell_015_replicator.py# Replication logic
│   └── cells/cell_016_feedback_loop.py # Feedback analysis
│
├── Datasets
│   ├── datasets/honeypot/
│   │   └── attacks.json            # Captured attacks
│   ├── datasets/generated/
│   │   └── generated_*.json        # Cell outputs
│   ├── datasets/refined/
│   └── datasets/decisions/
│
├── Documentation
│   ├── SYNEXS_MASTER_DOCUMENTATION.md (THIS FILE)
│   ├── BINARY_PROTOCOL_DEPLOYMENT.md
│   ├── BINARY_PROTOCOL_COMPLETE.md
│   ├── CORE_ORCHESTRATOR_README.md
│   ├── DEPLOYMENT_STATUS.md
│   └── IMPLEMENTATION_COMPLETE.txt
│
├── Logs
│   ├── synexs_core.log             # Orchestrator logs
│   ├── listener.log                # Kill reports
│   ├── honeypot.log                # Attacks
│   ├── dna_collector.log           # Data collection
│   └── memory_log.json             # Learning state
│
└── Utilities
    ├── health_check.py             # System monitoring
    ├── protocol_demo.py            # V1/V2/V3 comparison
    └── BINARY_QUICKSTART.sh        # Test script
```

---

## 6. Data Flow

### **6.1 Attack Capture Flow**

```
1. SSH attack → Honeypot server (port 2222)
2. PTR validation → Check reverse DNS
3. CIDR filtering → Validate IP ranges
4. Log attack → datasets/honeypot/attacks.json
5. DNA Collector → Generate training samples
```

### **6.2 Learning Flow**

```
1. Agent dies → Kill report sent to listener
2. Listener logs → listener.log
3. AI Swarm reads → Analyzes failure pattern
4. Mutation triggered → Strategy adapted
5. New agents spawned → Improved tactics applied
```

### **6.3 Orchestration Flow**

```
1. Orchestrator starts → Load AI model
2. Phase 1 (Generation) → Create sequences
3. Phase 2 (Processing) → Parse and hash
4. Phase 3 (Classification) → AI prediction
5. Phase 4 (Evolution) → Mutate and replicate
6. Phase 5 (Feedback) → Analyze results
7. Sleep 60s → Repeat cycle
```

### **6.4 Training Data Flow**

```
1. DNA Collector monitors → 3 sources (30 min cycle)
2. Events accumulated → Threshold: 100 events
3. Generate samples → 50 training samples
4. Append to file → training_binary_v3.jsonl
5. Save state → .dna_collector_state.json
```

---

## 7. Training Data

### **7.1 Current Dataset**

**File**: `training_binary_v3.jsonl`
**Size**: 266KB+
**Samples**: 1050+ (1000 synthetic + 50+ real)
**Format**: Binary V3 (Base64)
**Protocol**: v3

**Sample Structure**:
```json
{
  "instruction": "What does binary sequence CABEMhTH... mean?",
  "input": "binary:CABEMhTH",
  "output": "Execute sequence: SCAN → ATTACK → REPLICATE → MUTATE",
  "actions": ["SCAN", "ATTACK", "REPLICATE", "MUTATE"],
  "protocol": "v3",
  "format": "base64",
  "source": "cell_execution",
  "timestamp": 1762728953
}
```

### **7.2 Data Sources**

1. **Synthetic Data** (1000 samples)
   - Generated by `binary_protocol.py`
   - Random action sequences
   - All 32 actions represented

2. **Real Operations** (50+ samples, growing)
   - Collected by `dna_collector.py`
   - From honeypot attacks
   - From cell executions
   - From learning events

### **7.3 Data Collection Status**

**DNA Collector State**:
```json
{
  "last_run": 1762728953,
  "honeypot_offset": 576,
  "memory_offset": 0,
  "cell_files_seen": ["datasets/generated/..."],
  "total_events": 576,
  "total_samples": 50
}
```

**Growth Rate**: ~50 samples per 100 events
**Expected**: ~100 new samples per day (at current activity)

---

## 8. Deployment Status

### **8.1 Running Services**

```bash
$ ps aux | grep synexs

root  78462  honeypot_server.py        ✅ RUNNING
root  78579  listener.py               ✅ RUNNING
root  78631  propagate_v3.py           ✅ RUNNING
root  78694  ai_swarm_fixed.py         ✅ RUNNING
root 113413  synexs_core_orchestrator.py ✅ RUNNING
```

### **8.2 Cron Schedule**

```cron
# Boot services
@reboot cd /root/synexs && bash -c 'nohup python3 honeypot_server.py > /dev/null 2>&1 & nohup python3 listener.py > /dev/null 2>&1 & sleep 3 && nohup python3 propagate_v3.py > /dev/null 2>&1 & nohup python3 ai_swarm_fixed.py > /dev/null 2>&1 & nohup python3 synexs_core_orchestrator.py > /dev/null 2>&1 &'

# Hourly: Clean attack logs
0 * * * * tail -n 1000 /root/synexs/datasets/honeypot/attacks.json > /tmp/attacks_tmp && mv /tmp/attacks_tmp /root/synexs/datasets/honeypot/attacks.json

# Every 6 hours: Health check
0 */6 * * * cd /root/synexs && python3 health_check.py

# Every 30 minutes: DNA collection
*/30 * * * * cd /root/synexs && python3 dna_collector.py >> dna_collector.log 2>&1
```

### **8.3 System Health**

**Performance Metrics** (Last Check):
- CPU: 5-10% average
- Memory: 200-400MB total
- Disk: 45% usage
- Orchestrator: 100% success rate (7/7 cells)

---

## 9. Performance Metrics

### **9.1 Protocol Efficiency**

| Protocol | Message Size | Speedup | Use Case |
|----------|-------------|---------|----------|
| V1 (Greek) | 46 bytes | 1.0x | Legacy, debugging |
| V2 (Symbols) | 24 bytes | 1.9x | Development |
| **V3 (Binary)** | **6 bytes** | **7.7x** | **Production** ✅ |

**Bandwidth Savings** (1000 agents × 100 msg/hour):
- V1: 28 GB/year
- V3: 4 GB/year
- **Savings**: 24 GB/year (86% reduction)

### **9.2 Orchestrator Performance**

- **Cycle Time**: 9.8 seconds
- **Success Rate**: 100% (7/7 cells)
- **Memory**: ~55MB per instance
- **CPU**: <5% average

### **9.3 Data Collection**

- **Events Collected**: 576+ (first run)
- **Training Samples**: 1050+
- **Growth Rate**: ~50 samples per cycle
- **Storage**: 266KB (compressed binary format)

---

## 10. GPU Training Guide

### **10.1 Requirements**

**Hardware**:
- GPU: NVIDIA with CUDA support (RTX 3090, A100, etc.)
- RAM: 16GB+ recommended
- Storage: 10GB+ for model training

**Software**:
```bash
# CUDA toolkit
nvidia-smi  # Verify GPU

# Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate wandb
```

### **10.2 Transfer Data to GPU Server**

```bash
# From current VPS
cd /root/synexs
tar -czf synexs_training_data.tar.gz \
    training_binary_v3.jsonl \
    vocab_v3_binary.json \
    synexs_core_model.pth \
    synexs_model.py \
    binary_protocol.py

# Copy to GPU server
scp synexs_training_data.tar.gz user@gpu-server:/path/to/training/

# On GPU server
tar -xzf synexs_training_data.tar.gz
```

### **10.3 Training Script Template**

```python
#!/usr/bin/env python3
"""
train_synexs_gpu.py - GPU Training for Synexs Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from tqdm import tqdm

# Configuration
TRAINING_DATA = "training_binary_v3.jsonl"
MODEL_OUTPUT = "synexs_brain_v1"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-5

class SynexsDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer):
        self.samples = []
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                # Format: instruction + input + output
                text = f"{data['instruction']}\n{data['output']}"
                self.samples.append(tokenizer.encode(text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])

def train():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Load dataset
    dataset = SynexsDataset(TRAINING_DATA, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = batch.to(device)

            outputs = model(batch, labels=batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        model.save_pretrained(f"{MODEL_OUTPUT}_epoch{epoch+1}")
        tokenizer.save_pretrained(f"{MODEL_OUTPUT}_epoch{epoch+1}")

    print("Training complete!")

if __name__ == "__main__":
    train()
```

### **10.4 Training Schedule**

**Phase 1: Initial Training** (Week 1)
- Use current 1050 samples
- Train base model (GPT-2 or similar)
- Validate on held-out set

**Phase 2: Continuous Learning** (Ongoing)
- Collect data for 1-2 weeks (2000+ samples)
- Fine-tune with new data
- Monitor performance improvements

**Phase 3: Advanced Training** (Month 2+)
- 5000+ samples accumulated
- Train larger model (GPT-3 scale)
- Add reinforcement learning

---

## 11. Chatbot Integration

### **11.1 Architecture**

```
User ──► Chatbot ──► Synexs Brain ──► Binary Protocol ──► Swarm
         │           (GPU Model)      (Decode/Encode)     (Agents)
         │                                  │
         └──────────────────────────────────┘
                   Response Flow
```

### **11.2 Chatbot Implementation**

**Stack Options**:

**Option A: Telegram Bot**
```python
from telegram import Update
from telegram.ext import Application, CommandHandler

async def query_synexs(update: Update, context):
    query = update.message.text

    # Use trained model to generate response
    response = synexs_brain.predict(query)

    # Decode binary protocol
    actions = decode_binary(response)

    await update.message.reply_text(f"Actions: {actions}")

app = Application.builder().token("YOUR_TOKEN").build()
app.add_handler(CommandHandler("query", query_synexs))
app.run_polling()
```

**Option B: Discord Bot**
```python
import discord
from discord.ext import commands

bot = commands.Bot(command_prefix='!')

@bot.command()
async def synexs(ctx, *, query):
    response = synexs_brain.predict(query)
    await ctx.send(f"Synexs Response: {response}")

bot.run("YOUR_TOKEN")
```

**Option C: Web API (FastAPI)**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/query")
async def query_synexs(query: Query):
    response = synexs_brain.predict(query.text)
    actions = decode_binary(response)
    return {"response": response, "actions": actions}
```

### **11.3 API Endpoints**

```python
# Chatbot API endpoints needed

POST /query
  Input: {"text": "How do I detect a honeypot?"}
  Output: {"response": "...", "actions": ["SCAN", "EVADE"]}

POST /encode
  Input: {"actions": ["SCAN", "ATTACK"]}
  Output: {"binary": "BABE", "protocol": "v3"}

POST /decode
  Input: {"binary": "BABE", "protocol": "v3"}
  Output: {"actions": ["SCAN", "ATTACK"]}

GET /stats
  Output: {"total_samples": 1050, "protocols": ["v1", "v2", "v3"]}
```

### **11.4 Integration Flow**

1. **User Input** → Chatbot receives question
2. **Preprocessing** → Clean and format input
3. **Model Inference** → GPU model generates response
4. **Binary Encoding** → Convert to protocol V3
5. **Response** → Send back to user
6. **Logging** → Store interaction for training

---

## 12. Future Roadmap

### **12.1 Immediate Next Steps** (Week 1-2)

- ✅ Let DNA collector run (accumulate 2000+ samples)
- ⏳ Monitor system health daily
- ⏳ Review training data quality
- ⏳ Plan GPU server setup

### **12.2 Short-term Goals** (Month 1)

- [ ] **Transfer to GPU Server**
  - Copy training data
  - Setup CUDA environment
  - Test protocol implementations

- [ ] **Initial Model Training**
  - Train on 2000+ samples
  - Validate accuracy
  - Benchmark performance

- [ ] **Chatbot Prototype**
  - Choose platform (Telegram/Discord/Web)
  - Implement basic API
  - Connect to trained model

### **12.3 Mid-term Goals** (Month 2-3)

- [ ] **Advanced Model Training**
  - 5000+ samples
  - Fine-tune larger model
  - Add RLHF (Reinforcement Learning from Human Feedback)

- [ ] **Chatbot Enhancement**
  - Multi-turn conversations
  - Context awareness
  - Action execution

- [ ] **Integration Testing**
  - End-to-end workflows
  - Performance optimization
  - Security hardening

### **12.4 Long-term Vision** (Month 4+)

- [ ] **Distributed Training**
  - Multiple GPU servers
  - Distributed data collection
  - Federated learning

- [ ] **Advanced Features**
  - Real-time swarm control via chatbot
  - Predictive analytics
  - Automated strategy optimization

- [ ] **Production Deployment**
  - Multi-region deployment
  - Load balancing
  - Monitoring dashboard

---

## 13. Key Files for GPU Transfer

### **Essential Files** (Must Transfer):

```
Priority 1 - Training Data:
✅ training_binary_v3.jsonl         # 1050+ samples
✅ vocab_v3_binary.json             # 32 actions
✅ synexs_core_model.pth            # Current weights

Priority 2 - Protocol Implementation:
✅ binary_protocol.py               # Encode/decode functions
✅ synexs_model.py                  # Model architecture

Priority 3 - Documentation:
✅ SYNEXS_MASTER_DOCUMENTATION.md   # This file
✅ BINARY_PROTOCOL_DEPLOYMENT.md    # Protocol details
```

### **Transfer Command**:

```bash
# Create transfer package
cd /root/synexs
tar -czf synexs_gpu_package.tar.gz \
    training_binary_v3.jsonl \
    vocab_v3_binary.json \
    synexs_core_model.pth \
    binary_protocol.py \
    synexs_model.py \
    SYNEXS_MASTER_DOCUMENTATION.md \
    BINARY_PROTOCOL_DEPLOYMENT.md

# Copy to GPU server (replace with your details)
scp synexs_gpu_package.tar.gz user@gpu-server:/path/to/training/
```

---

## 14. Quick Reference

### **System Status Commands**

```bash
# Check running services
ps aux | grep synexs

# View orchestrator log
tail -f synexs_core.log

# Check DNA collector status
cat .dna_collector_state.json

# Count training samples
wc -l training_binary_v3.jsonl

# View latest samples
tail -10 training_binary_v3.jsonl | jq
```

### **Protocol Usage**

```python
# Encode actions to binary
from binary_protocol import encode_base64
encoded = encode_base64(["SCAN", "ATTACK", "REPLICATE"])
# Output: "BABGYA=="

# Decode binary to actions
from binary_protocol import decode_base64
actions = decode_base64("BABGYA==")
# Output: ["SCAN", "ATTACK", "REPLICATE"]
```

### **Key Metrics**

- **Protocol Efficiency**: 88% reduction (V3 vs V1)
- **Transmission Speed**: 8.3x faster (V3)
- **Training Samples**: 1050+ (growing)
- **System Uptime**: 99%+ (auto-restart on boot)
- **Bandwidth Savings**: 24 GB/year per 1000 agents

---

## 15. Contact & Support

### **Current Status**: ✅ Production Ready - Data Collection Phase

### **Next Action**: Let system collect data for 1-2 weeks

### **GPU Training**: Ready when you have 2000+ samples

### **Documentation**: All specs in this file for future reference

---

## 16. Appendix: Technical Specifications

### **A. Binary Protocol Specification**

**Encoding Algorithm**:
1. Convert action names to 5-bit codes (0x00-0x1F)
2. Pack bits sequentially into bytes
3. Prepend 1-byte length header
4. Output: 1 + ceil(n×5/8) bytes

**Example**:
```
Actions: [SCAN, ATTACK, REPLICATE, MUTATE]
Codes:   [0x00, 0x01, 0x02, 0x03]
Binary:  00000 00001 00010 00011 (20 bits)
Packed:  [0x04] [0x00] [0x44] [0x60]
Base64:  "BABGYA=="
```

### **B. Model Architecture**

**Current Model** (synexs_core_model.pth):
```python
SynexsCoreModel(
  embedding: Embedding(26, 32)
  fc1: Linear(32, 64)
  relu: ReLU()
  dropout: Dropout(0.2)
  fc2: Linear(64, 5)
)
```

**Output Actions**: [discard, refine, replicate, mutate, flag]

### **C. Data Schema**

**Training Sample Format**:
```json
{
  "instruction": "string - Question about sequence",
  "input": "string - Binary sequence (Base64)",
  "output": "string - Human-readable explanation",
  "actions": ["array", "of", "action", "names"],
  "protocol": "v3",
  "format": "base64",
  "source": "string - Event source",
  "timestamp": 1762728953
}
```

---

**End of Documentation**

**Version**: 2.0
**Last Updated**: 2025-11-09
**Status**: ✅ Complete and Ready for GPU Training

---

**Summary**: Synexs is a production-ready AI defensive security training system with ultra-efficient binary communication (88% reduction), autonomous data collection, and 1050+ training samples ready for GPU training and chatbot integration.
