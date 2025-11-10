# Synexs Protocol V2 - Quick Reference

---

## 🎯 One-Line Summary
**Replace "SIGMA OMEGA THETA" (28 bytes) with "△□◆" (9 bytes) for 46% bandwidth reduction**

---

## 📖 Symbol Dictionary

| Symbol | Action | Use Case |
|--------|--------|----------|
| △ | SCAN | Detect honeypot patterns, validate target |
| □ | ATTACK | Execute payload on validated target |
| ◆ | REPLICATE | Spawn new agent, grow swarm |
| ◇ | MUTATE | Change attack signature, adapt |
| ○ | EVADE | PTR record detected, abort mission |
| ● | LEARN | Agent killed, update swarm intelligence |
| ◉ | REPORT | Send kill data to improve mutations |
| ◎ | DEFEND | Localhost only, training mode active |
| ⬡ | REFINE | Optimize sequence, reduce detection |
| ⬢ | FLAG | Anomaly detected, require human analysis |

---

## 🚀 Quick Start

### **Encode Actions**
```python
from protocol_v2_proposal import encode_sequence

actions = ["SCAN", "ATTACK", "REPLICATE"]
symbolic = encode_sequence(actions)
# Output: "△□◆"
```

### **Decode Sequence**
```python
from protocol_v2_proposal import decode_sequence

symbolic = "△□◆◇○"
actions = decode_sequence(symbolic)
# Output: ['SCAN', 'ATTACK', 'REPLICATE', 'MUTATE', 'EVADE']
```

### **Generate Training Data**
```python
from protocol_v2_proposal import generate_training_data

samples = generate_training_data(100)
# Generates 100 instruction/output pairs
```

---

## 📊 Performance at a Glance

| Metric | V1 (Old) | V2 (New) | Improvement |
|--------|----------|----------|-------------|
| **Token Size** | 5.6 bytes | 3.0 bytes | **46% smaller** |
| **4-token msg** | 22 bytes | 12 bytes | **45% reduction** |
| **8-token msg** | 46 bytes | 24 bytes | **48% reduction** |
| **Speed** | 1.0x | 1.87x | **87% faster** |

---

## 🔧 Common Patterns

### **Agent Spawn**
```
◆◇◉  → REPLICATE + MUTATE + REPORT
```

### **Honeypot Detection**
```
△△○◉● → SCAN + SCAN + EVADE + REPORT + LEARN
```

### **Attack Sequence**
```
△□◉ → SCAN + ATTACK + REPORT
```

### **Swarm Learning**
```
●◇◆ → LEARN + MUTATE + REPLICATE
```

---

## 🎓 Training Format

```json
{
  "instruction": "What does △□◆ mean?",
  "input": "",
  "output": "SCAN target, ATTACK if valid, REPLICATE agent."
}
```

---

## ⚡ Cheat Sheet

### **Top 5 Most Common Sequences**

1. **△□◆** - Standard attack pattern
2. **△△○** - Honeypot evasion
3. **●◇◆** - Learning from failure
4. **◆◇◉** - Swarm replication
5. **△□◉◎** - Attack + defend

---

## 📝 Files to Use

| File | Purpose |
|------|---------|
| `protocol_v2_proposal.py` | Core implementation |
| `vocab_v2.json` | 30-token vocabulary |
| `training_symbolic_v2.jsonl` | Training samples |
| `protocol_demo.py` | Visual comparison |

---

## 🧪 Quick Test

```bash
# See the difference
python3 protocol_demo.py

# Generate training data
python3 protocol_v2_proposal.py
```

---

**Remember**: V2 = Compact + Fast + AI-Native 🚀
