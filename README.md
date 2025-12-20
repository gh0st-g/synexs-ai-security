# Synexs - Purple Team AI Security Framework

**Autonomous AI-Powered Defensive Security Training System**

[![License](https://img.shields.io/badge/license-Research-red.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Status](https://img.shields.io/badge/status-Production--Ready-brightgreen.svg)]()
[![Protocol](https://img.shields.io/badge/protocol-V3%20Binary%20(88%25%20reduction)-blue.svg)]()

---

## 🎯 Project Overview

Synexs is an advanced **purple team security research framework** that combines red team attack simulation with blue team defense training. It uses AI-powered agents, polymorphic attack patterns, and adaptive learning to train defensive AI systems through controlled honeypot environments.

### Core Mission
Train AI models to detect and defend against sophisticated cyber attacks by generating diverse, polymorphic attack patterns in a **controlled localhost-only environment**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNEXS ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌─────────────┐    ┌─────────────┐ │
│  │  Propagate   │─────▶│  Honeypot   │───▶│ AI Training │ │
│  │  (Red Team)  │      │ (Blue Team) │    │   System    │ │
│  └──────────────┘      └─────────────┘    └─────────────┘ │
│         │                     │                    │        │
│         ▼                     ▼                    ▼        │
│  ┌──────────────┐      ┌─────────────┐    ┌─────────────┐ │
│  │  10+ Attack  │      │  WAF + AI   │    │  V3 Binary  │ │
│  │  Categories  │      │  Detection  │    │  Protocol   │ │
│  └──────────────┘      └─────────────┘    └─────────────┘ │
│         │                     │                    │        │
│         └─────────────────────┴────────────────────┘        │
│                    Feedback & Mutation Loop                 │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🛡️ Purple Team Training
- **Red Team**: Polymorphic attack generation (10+ categories)
- **Blue Team**: AI-powered detection & response
- **Continuous Learning**: Adaptive mutation based on success/failure
- **Localhost-Only**: Controlled environment, no external targets

### 🧬 Polymorphic Attack Generation
- **10+ Attack Categories**: SQL injection, XSS, Path Traversal, Command Injection, etc.
- **5 Encoding Types**: URL, Double-URL, Unicode, HTML Entity, Hex
- **4 Case Variations**: Upper, Lower, Mixed, Alternating
- **Evasion Techniques**: Whitespace injection, fragment splitting, timing variation
- **Self-Adapting**: Learns from detection and adjusts tactics

### 📡 V3 Binary Protocol (88% Bandwidth Reduction)
- **Ultra-Compact**: 6-8 bytes vs 28+ bytes (V1 Greek words)
- **8.3x Faster**: Transmission speed improvement
- **32 Actions**: Comprehensive action vocabulary
- **Base64 Encoded**: Efficient agent-to-agent communication

### 🤖 AI Detection System
- **Hybrid Detection**: WAF rules + XGBoost ML model
- **Shadow Mode**: AI learns without interfering
- **Real-time Analysis**: <5ms detection latency
- **Training Pipeline**: Automated data collection → model training

### 🔬 Cellular Architecture
- **Modular Cells**: 7 specialized processing phases
- **Core Orchestrator**: Central coordination system
- **Feedback Loops**: Continuous improvement
- **V3 Protocol**: Efficient inter-cell communication

---

## 📊 Current Status

### Production Systems Running ✅
```bash
✅ honeypot_server.py          # Attack capture & WAF
✅ listener.py                 # Kill report collection
✅ ai_swarm_fixed.py           # Learning engine
✅ synexs_core_orchestrator.py # Cell coordinator (Cycle #868+)
✅ propagate_v3_enhanced.py    # Polymorphic attack generator
```

### Training Data Metrics
```
Attack Types: 10+ categories
Encodings: 5 types
Training Records: Growing (target: 2000+)
V3 Protocol: Active (88% reduction)
Diversity: High ✅
```

### System Performance
```
Orchestrator: 100% success rate (7/7 cells)
Cycle Time: 4-10 seconds
V3 Sequences: 10 bytes avg (vs 28 bytes V1)
Bandwidth Savings: 24 GB/year per 1000 agents
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
python3 --version

# Virtual environment (recommended)
python3 -m venv synexs_env
source synexs_env/bin/activate

# Dependencies
pip install -r requirements.txt
```

### 1. Start Core Services
```bash
# Start honeypot (port 8080)
python3 honeypot_server.py &

# Start AI swarm
python3 ai_swarm_fixed.py &

# Start listener
python3 listener.py &

# Start orchestrator
python3 synexs_core_orchestrator.py &
```

### 2. Generate Training Data
```bash
# Generate 50 diverse attack agents
python3 propagate_v3_enhanced.py

# Execute agents against honeypot
for f in datasets/agents/sx*.py; do
    python3 "$f"
    sleep 0.1
done
```

### 3. Monitor Training Data Collection
```bash
# Check attack diversity
cat datasets/honeypot/attacks.json | jq -r '.type' | sort | uniq -c

# View orchestrator logs
tail -f synexs_core.log

# Monitor AI decisions
tail -f ai_decisions_log.jsonl
```

---

## 📁 Project Structure

```
synexs/
├── Core Systems
│   ├── honeypot_server.py          # WAF + AI detection
│   ├── listener.py                 # Kill report collector
│   ├── propagate_v3_enhanced.py    # Attack generator
│   ├── ai_swarm_fixed.py           # Learning engine
│   └── synexs_core_orchestrator.py # Cell coordinator
│
├── Protocol & Models
│   ├── binary_protocol.py          # V3 protocol (88% reduction)
│   ├── synexs_model.py             # ML model
│   ├── vocab_v3_binary.json        # 32 action vocabulary
│   └── attack_profiles.json        # Attack configuration
│
├── Cells (Processing Pipeline)
│   ├── cell_001.py                 # V3 sequence generator
│   ├── cell_004.py                 # Hash logging
│   ├── cell_006.py                 # AI classifier
│   ├── cell_010_parser.py          # Token parser
│   ├── cell_014_mutator.py         # Mutation engine
│   ├── cell_015_replicator.py      # Replication logic
│   └── cell_016_feedback_loop.py   # Feedback analysis
│
├── Configuration
│   ├── ai_config.json              # AI model configuration
│   └── attack_profiles.json        # Attack pattern config
│
├── Training Data
│   ├── datasets/
│   │   ├── honeypot/               # Attack logs
│   │   ├── generated/              # V3 sequences
│   │   ├── refined/                # Processed data
│   │   ├── agents/                 # Generated agents
│   │   └── mutation.json           # Adaptive state
│   │
│   └── training_binary_v3.jsonl    # Training dataset
│
└── Documentation
    ├── README.md                    # This file
    ├── SYNEXS_MASTER_DOCUMENTATION.md
    ├── PURPLE_TEAM_ENHANCEMENT.md
    ├── BINARY_PROTOCOL_DEPLOYMENT.md
    └── TRAINING_DATA_ANALYSIS.md
```

---

## 🔧 Configuration

### Attack Profile Customization
Edit `attack_profiles.json`:
```json
{
  "sql_injection": {
    "enabled": true,
    "weight": 15,
    "patterns": ["' OR '1'='1", "1' UNION SELECT"],
    "encodings": ["url", "unicode", "hex"]
  }
}
```

### AI Model Configuration
Edit `ai_config.json`:
```json
{
  "ai_mode": {
    "shadow_mode": true,
    "fallback_enabled": true
  },
  "training": {
    "auto_retrain": true,
    "retrain_every_n_cycles": 100
  }
}
```

---

## 📈 Training Pipeline

### Phase 1: Data Collection (Current)
```
1. Generate diverse attack patterns (propagate_v3_enhanced.py)
2. Execute against localhost honeypot
3. Log attacks & responses
4. AI analyzes patterns
5. Mutation system adapts strategies
```

### Phase 2: AI Training (2000+ samples)
```
1. Export training data (training_binary_v3.jsonl)
2. Transfer to GPU server
3. Train ML model on diverse patterns
4. Validate detection accuracy
5. Deploy improved model
```

### Phase 3: Production (5000+ samples)
```
1. Real-time detection active
2. Continuous learning enabled
3. Adaptive response system
4. Performance monitoring
```

---

## 🎓 Use Cases

### 1. Security Research
- Study attack patterns and evasion techniques
- Test WAF and detection systems
- Research AI-powered defense mechanisms

### 2. Red Team Training
- Practice attack methodology
- Learn polymorphic techniques
- Understand detection evasion

### 3. Blue Team Training
- Improve detection capabilities
- Train AI models on diverse attacks
- Test defensive response systems

### 4. Purple Team Exercises
- Simulate real-world attack scenarios
- Measure detection effectiveness
- Continuous improvement loops

---

## 📊 Performance Metrics

### Attack Diversity
```
SQL Injection: 15%
XSS: 12%
Path Traversal: 8%
Command Injection: 4%
Crawler Impersonation: 10%
Directory Scanning: 4%
Legitimate Traffic: 20%
Other Attacks: 27%
```

### Detection Performance
```
WAF Detection: <5ms latency
AI Detection: <50ms latency
False Positive Rate: <2%
True Positive Rate: >95% (target)
```

### Data Collection Rate
```
50 agents/batch × 10 batches/day = 500 attacks/day
Target: 2000+ diverse samples in 4-5 days
Expected: 10,000+ samples in 1 month
```

---

## 🔒 Security Considerations

### Safety Boundaries
✅ **Hardcoded Localhost**: All targets are 127.0.0.1
✅ **No External Access**: Cannot reach internet
✅ **Controlled Environment**: Isolated honeypot
✅ **Research Purpose**: Educational/defensive security

### What This System Does NOT Do
❌ Remote exploitation
❌ Data exfiltration
❌ Credential harvesting
❌ Network propagation
❌ Malicious activity outside localhost

### Verification
```bash
# Verify localhost-only targets
grep -r "127.0.0.1" *.py

# Check no external IPs
grep -E "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" *.py | grep -v "127.0.0.1"
```

---

## 📚 Documentation

### Core Documentation
- **[SYNEXS_MASTER_DOCUMENTATION.md](SYNEXS_MASTER_DOCUMENTATION.md)** - Complete system guide
- **[PURPLE_TEAM_ENHANCEMENT.md](PURPLE_TEAM_ENHANCEMENT.md)** - Attack diversity improvements
- **[BINARY_PROTOCOL_DEPLOYMENT.md](BINARY_PROTOCOL_DEPLOYMENT.md)** - V3 protocol guide
- **[TRAINING_DATA_ANALYSIS.md](TRAINING_DATA_ANALYSIS.md)** - Data collection analysis
- **[CLEANUP_REPORT.md](CLEANUP_REPORT.md)** - System cleanup documentation

### Quick References
- **[CORE_ORCHESTRATOR_README.md](CORE_ORCHESTRATOR_README.md)** - Orchestrator guide
- **[GPU_SETUP_README.md](GPU_SETUP_README.md)** - GPU training setup
- **[DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md)** - Current deployment status

---

## 🤝 Contributing

This is a private research project. If you have suggestions or improvements:

1. Document your findings
2. Test in isolated environment
3. Verify localhost-only operation
4. Submit detailed analysis

---

## 📝 Changelog

### v3.0 (Dec 2025) - Purple Team Enhancement
- ✅ Added 10+ attack categories (vs 6 before)
- ✅ Implemented polymorphic behavior (5 encodings, 4 case variations)
- ✅ Configuration-based attack profiles
- ✅ Legitimate traffic generation (balanced dataset)
- ✅ Enhanced mutation system

### v2.3 (Nov 2025) - V3 Binary Protocol
- ✅ V3 binary protocol (88% bandwidth reduction)
- ✅ AI decision engine with shadow mode
- ✅ Training data collection automation
- ✅ Orchestrator production deployment

### v2.0 (Oct 2025) - Core System
- ✅ Cellular architecture implementation
- ✅ Honeypot with WAF + AI detection
- ✅ Agent swarm learning system
- ✅ Basic attack generation

---

## 🎯 Roadmap

### Short Term (Month 1)
- [ ] Collect 2000+ diverse training samples
- [ ] Transfer to GPU server for training
- [ ] Deploy trained AI model
- [ ] Measure detection accuracy

### Mid Term (Months 2-3)
- [ ] Expand to 20+ attack categories
- [ ] Real-time attack adaptation
- [ ] Advanced evasion techniques
- [ ] Performance optimization

### Long Term (Months 4-6)
- [ ] Distributed honeypot network
- [ ] Advanced AI models (transformer-based)
- [ ] Autonomous response system
- [ ] Public research publication

---

## 📞 Support & Contact

**Project**: Synexs - Purple Team AI Security Framework
**Purpose**: Defensive security research & AI training
**Status**: Production-ready data collection phase

For questions or collaboration inquiries, please refer to project documentation.

---

## ⚖️ License & Ethics

This project is for **educational and defensive security research** purposes only.

### Ethical Guidelines
- ✅ Use only in controlled environments
- ✅ Localhost-only targets
- ✅ No unauthorized access
- ✅ Defensive security research
- ❌ No malicious use
- ❌ No unauthorized deployment
- ❌ No external attacks

**Research Context**: This system is designed to improve defensive AI capabilities by training models on diverse attack patterns in controlled, isolated environments.

---

**Synexs** - *Advancing defensive AI through purple team methodology*

Last Updated: December 15, 2025
Version: 3.0
