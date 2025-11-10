# Synexs Phase 1 - Quick Start Guide

**Multi-Agent Team Coordination & GPU Training Pipeline**

---

## What is Phase 1?

Phase 1 implements coordinated teams of AI agents that work together on complex missions. The system:
- Simulates 5-agent teams with specialized roles
- Logs all communications and decisions for training
- Formats data for GPU-accelerated neural network training
- Trains models to predict mission outcomes

---

## Quick Start (5 minutes)

### 1. Test the Team Simulator

```bash
# Run basic team simulation test
python3 synexs_team_simulator.py
```

This will:
- Create a 5-agent team (Scout, Analyzer, Executor, Exfiltrator, Cleaner)
- Run 2 test missions (easy and hard scenarios)
- Show agent communications and decisions
- Display team performance metrics

### 2. Run Training Data Collection

```bash
# Quick test: 10 missions
python3 synexs_phase1_runner.py --quick

# Full training: 100 missions
python3 synexs_phase1_runner.py --missions 100

# Large dataset: 1000 missions
python3 synexs_phase1_runner.py --missions 1000 --output ./large_training
```

This will:
- Generate diverse mission scenarios
- Execute missions with full team coordination
- Log all data in GPU-ready format
- Create training batches automatically

### 3. Train Neural Network on GPU

```bash
# Train model on collected data
python3 synexs_gpu_trainer.py ./training_logs/batches

# Or specify custom settings
python3 synexs_gpu_trainer.py ./training_logs/batches --epochs 50 --batch-size 32
```

This will:
- Load batched training data
- Train LSTM-based mission predictor
- Use GPU if available (CUDA)
- Save trained model
- Generate training report

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 PHASE 1 PIPELINE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Mission Generator → Team Simulator → Training Logger  │
│         ↓                  ↓                 ↓          │
│  [Environments]    [Agent Teams]     [GPU Batches]     │
│         ↓                  ↓                 ↓          │
│   - Easy (20%)      5 Agents:        PyTorch Format    │
│   - Medium (50%)    • Scout          [B, S, F]         │
│   - Hard (20%)      • Analyzer       B = Batch         │
│   - Honeypot (10%)  • Executor       S = Sequence      │
│                     • Exfiltrator    F = Features      │
│                     • Cleaner                           │
│                            ↓                            │
│                     Communications                      │
│                     + Decisions                         │
│                            ↓                            │
│                     ┌──────────────┐                    │
│                     │  GPU Trainer │                    │
│                     └──────┬───────┘                    │
│                            ↓                            │
│                  Trained Mission Predictor              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Team Simulator (`synexs_team_simulator.py`)

**Agent Roles:**
- **Scout** - Network reconnaissance and host discovery
- **Analyzer** - Security assessment and honeypot detection
- **Executor** - Target exploitation and execution
- **Exfiltrator** - Data retrieval and extraction
- **Cleaner** - Trace removal and cleanup

**Mission Flow:**
1. Scout gathers intelligence → Sends to Analyzer
2. Analyzer assesses risk → Makes go/no-go decision
3. If go: Executor engages targets
4. Exfiltrator retrieves data
5. Cleaner removes traces

**Logged Data:**
- All inter-agent messages (binary protocol)
- Decision rationale and confidence
- Environment observations
- Performance metrics

### 2. Training Logger (`synexs_training_logger.py`)

**Features:**
- Real-time data capture
- GPU-optimized tensor formatting
- Automatic batch creation (default: 32 missions per batch)
- PyTorch DataLoader integration

**Output Format:**
```python
{
    'features': torch.Tensor,  # [batch, sequence, features]
    'labels': torch.Tensor,    # [batch] (0=success, 1=fail, 2=abort)
    'metadata': List[Dict]     # Mission details
}
```

### 3. GPU Trainer (`synexs_gpu_trainer.py`)

**Model Architecture:**
- LSTM layers for sequence processing
- Attention mechanism (focuses on important messages)
- Fully connected classification layers
- 3-class output: SUCCESS / FAILURE / ABORTED

**Training Features:**
- Automatic GPU detection (CUDA)
- Early stopping to prevent overfitting
- Train/validation split (80/20)
- Model checkpointing

### 4. Phase 1 Runner (`synexs_phase1_runner.py`)

**Mission Generator:**
- Diverse difficulty levels (easy → honeypot)
- Realistic environment simulation
- Dynamic risk parameters
- Honeypot indicators

**Training Session:**
- Configurable mission count
- Progress tracking
- Performance statistics
- Automatic data export

---

## Training Data Format

### Mission Log (JSONL)
```json
{
  "mission_id": "a7f3d2b1",
  "timestamp": 1699564800,
  "duration": 47.3,
  "team_composition": ["scout", "analyzer", "executor", "exfiltrator", "cleaner"],
  "environment": {
    "type": "medium_network",
    "risk_level": 0.45,
    "success_probability": 0.67,
    "detection_likelihood": 0.32,
    "honeypot_signals": ["timing"],
    "defenses": ["firewall", "IDS"]
  },
  "communications": [
    {
      "message_id": "m001",
      "sender": "scout_a7f3",
      "receiver": "analyzer_b2c4",
      "protocol": "binary_v2",
      "size_bytes": 47,
      "latency_ms": 12.3,
      "information_value": 0.87
    }
  ],
  "decisions": [
    {
      "decision_id": "d001",
      "agent": "analyzer_b2c4",
      "decision_type": "proceed",
      "rationale": "Honeypot probability: 0.12",
      "confidence": 0.92
    }
  ],
  "metrics": {
    "efficiency": 0.94,
    "coordination_score": 0.89,
    "communication_overhead": 0.12
  },
  "status": "success"
}
```

### GPU Batch (PyTorch)
```python
# Shape: [batch_size, sequence_length, feature_dim]
features = torch.Tensor([32, 50, 19])

# Shape: [batch_size]
labels = torch.Tensor([32])  # 0/1/2

# Metadata
metadata = [
    {'mission_id': 'abc', 'efficiency': 0.94},
    # ... 31 more
]
```

---

## Performance Metrics

### Team Performance
- **Success Rate**: % of successful missions
- **Coordination Score**: Quality of inter-agent communication
- **Efficiency**: Overall mission performance (0-1)
- **Average Duration**: Time to complete missions

### Communication Metrics
- **Overhead**: Bandwidth usage vs payload
- **Latency**: Message transmission time
- **Information Value**: Usefulness of messages (0-1)
- **Protocol Efficiency**: Binary vs JSON compression

### Model Training
- **Training Accuracy**: Performance on training set
- **Validation Accuracy**: Performance on held-out data
- **Loss**: Cross-entropy loss (lower is better)
- **Convergence**: Epochs to reach optimal performance

---

## Example Usage

### Scenario 1: Quick Test
```bash
# Test all components
python3 synexs_team_simulator.py
python3 synexs_phase1_runner.py --quick
python3 synexs_gpu_trainer.py ./training_logs/batches
```

### Scenario 2: Generate Large Training Dataset
```bash
# Run 5000 missions
python3 synexs_phase1_runner.py \
    --missions 5000 \
    --output ./massive_training \
    --team-id alpha_team

# This will create ~156 batches (5000 / 32)
```

### Scenario 3: Continuous Training Loop
```bash
# Generate data in background
python3 synexs_phase1_runner.py --missions 10000 --output ./continuous_training &

# Train model as data arrives
watch -n 300 'python3 synexs_gpu_trainer.py ./continuous_training/batches'
```

---

## Requirements

### Software
- Python 3.8+
- PyTorch 2.0+
- NumPy
- CUDA (optional, for GPU acceleration)

### Hardware Recommendations
- **CPU Training**: Any modern CPU, ~2 hours per 1000 missions
- **GPU Training**: GTX 1080+ or better, ~15 minutes per 1000 missions
- **Storage**: ~10MB per 100 missions (~100GB for 1M missions)
- **Memory**: 8GB+ RAM (16GB+ recommended for large datasets)

### Installation
```bash
# Install dependencies
pip install torch numpy

# Or use requirements
pip install -r requirements.txt
```

---

## Troubleshooting

### "No batch files found"
- Make sure you ran `synexs_phase1_runner.py` first
- Check that `--output` directory contains a `batches/` subdirectory
- Need at least 32 missions to create first batch

### "CUDA out of memory"
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Close other GPU applications
- Train on CPU if necessary (automatically falls back)

### Slow training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`
- Reduce sequence length if needed (edit `synexs_training_logger.py`)

### Low model accuracy
- Need more training data (run more missions)
- Increase model complexity (edit `MissionPredictor` architecture)
- Adjust learning rate or epochs
- Check for data imbalance (too many of one class)

---

## Next Steps

After completing Phase 1:

1. **Analyze Model Performance**
   - Review training reports
   - Test model predictions
   - Identify failure patterns

2. **Scale Up Data Collection**
   - Run 10,000+ missions
   - Vary environment difficulty distribution
   - Add more mission types

3. **Prepare for Phase 2**
   - Implement environment analysis engine
   - Add autonomous decision-making
   - Build experience database

4. **Optimize Performance**
   - Profile code for bottlenecks
   - Parallelize mission execution
   - Implement distributed training

---

## Phase 1 Success Criteria

- [x] Team simulator operational with 5 agent types
- [ ] 10,000+ logged mission executions
- [ ] Communication protocol efficiency > 85%
- [ ] GPU training pipeline processing real-time data
- [ ] Mission success prediction accuracy > 85%
- [ ] Model converges in < 50 epochs

---

## Questions or Issues?

See the main documentation:
- **Full Roadmap**: `SYNEXS_EVOLUTION_ROADMAP.md`
- **System Architecture**: `SYNEXS_MASTER_DOCUMENTATION.md`
- **GitHub**: https://github.com/gh0st-g/synexs

---

**Phase 1 Status**: ✅ Implementation Complete - Ready for Testing
**Next Phase**: Phase 2 - Autonomous Decision Making
**Estimated Timeline**: 6 months to Phase 2
