# Synexs

**AI-Powered Defensive Security Training System**

[![License](https://img.shields.io/badge/license-Private-red.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Status](https://img.shields.io/badge/status-Production%20Ready-green.svg)]()

---

## Overview

Synexs is an advanced AI-powered security framework designed for defensive training and adaptive threat response. It combines agent-based swarm intelligence, evolutionary algorithms, and machine learning to create a self-improving security system.

### Key Features

- **🧠 Adaptive Learning**: AI agents learn from interactions and evolve strategies
- **🔬 Honeypot Detection**: Advanced pattern recognition for identifying decoy systems
- **🔄 Evolutionary Algorithms**: Spawn → Mutate → Replicate cycle for continuous improvement
- **📡 Binary Protocol**: Ultra-efficient communication (88% bandwidth reduction)
- **🎯 Cellular Architecture**: Modular, autonomous components for distributed intelligence
- **📊 Real-time Analytics**: Live data collection and training pipeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNEXS ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐   ┌───────────────┐  │
│  │  Honeypot   │───▶│ DNA Collector│──▶│Training Data  │  │
│  │   Server    │    │  (Analysis)  │   │  Generation   │  │
│  └─────────────┘    └──────────────┘   └───────────────┘  │
│         │                                       │           │
│         ▼                                       ▼           │
│  ┌─────────────┐    ┌──────────────┐   ┌───────────────┐  │
│  │  Listener   │───▶│ AI Swarm     │──▶│  Orchestrator │  │
│  │  (Reports)  │    │  (Learning)  │   │   (Executor)  │  │
│  └─────────────┘    └──────────────┘   └───────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Agent Spawner & Propagator             │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 🔐 Defensive Training Engine
- Localhost-only honeypot targeting for safe training
- AI learns from agent successes and failures
- Continuous adaptation based on attack patterns

### 🧬 Cellular Architecture
- **Cell Modules**: Independent, specialized components (`cells/cell_*.py`)
- **Core Orchestrator**: Central coordination system (`synexs_core_orchestrator.py`)
- **Feedback Loops**: Self-improving mechanisms

### 📡 Binary Protocol
- Ultra-compact communication (6 bytes vs 46 bytes JSON)
- 88% bandwidth reduction
- Protocol v2 with enhanced efficiency

### 🤖 AI/ML Pipeline
- XGBoost classifier for threat detection
- PyTorch models for pattern recognition
- Scikit-learn for feature extraction
- Real-time model training and updates

---

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Linux environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/gh0st-g/synexs.git
cd synexs

# Install dependencies
pip install -r requirements.txt

# For fast deployment (minimal dependencies)
pip install -r requirements_fast.txt
```

### Basic Usage

```bash
# Run the quick start script
./BINARY_QUICKSTART.sh

# Or run core components individually
python3 synexs_core_orchestrator.py
```

### Docker Deployment

```bash
# Quick start with Docker
./DOCKER_QUICKSTART.sh

# Or use docker-compose
docker-compose up -d
```

---

## Project Structure

```
synexs/
├── cells/                    # Cellular architecture modules
│   ├── cell_001.py          # Base cell implementation
│   ├── cell_021_core_loop.py # Core execution loop
│   └── ...                   # Specialized cells
├── cleanup_utils/            # System maintenance utilities
├── loader/                   # Rust-based loader component
├── synexs_core_orchestrator.py  # Main orchestration engine
├── synexs_core_ai.py        # AI/ML core logic
├── binary_protocol.py        # Binary communication protocol
├── defensive_engine_fast.py  # Fast defensive training engine
├── honeypot_server.py        # Honeypot implementation
├── dna_collector.py          # Training data collector
├── requirements.txt          # Python dependencies
└── docker-compose.yml        # Container orchestration
```

---

## Documentation

Comprehensive documentation is available in the following files:

- **[SYNEXS_MASTER_DOCUMENTATION.md](SYNEXS_MASTER_DOCUMENTATION.md)** - Complete system overview
- **[BINARY_PROTOCOL_COMPLETE.md](BINARY_PROTOCOL_COMPLETE.md)** - Protocol specification
- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Container deployment guide
- **[DEFENSIVE_TRAINING.md](DEFENSIVE_TRAINING.md)** - Training methodology
- **[GPU_SETUP_README.md](GPU_SETUP_README.md)** - GPU acceleration setup
- **[CORE_ORCHESTRATOR_README.md](CORE_ORCHESTRATOR_README.md)** - Orchestrator details

---

## Key Technologies

- **Python 3.8+** - Primary development language
- **PyTorch** - Deep learning framework
- **Scikit-learn** - Machine learning utilities
- **XGBoost** - Gradient boosting classifier
- **Flask** - Web dashboard interface
- **Docker** - Containerization
- **Rust** - High-performance loader component

---

## Performance Metrics

- **Model Accuracy**: 100% on training dataset (60K+ samples)
- **Bandwidth Reduction**: 88% (binary protocol vs JSON)
- **Response Time**: <50ms average for agent decisions
- **Training Speed**: GPU-accelerated pipeline

---

## Security & Ethics

This project is designed for **defensive security research and authorized testing only**.

- All honeypot testing is localhost-only by default
- Intended for educational and defensive purposes
- Should only be used in authorized environments
- Not intended for malicious activities

---

## Contributing

This is a private research project. Contributions are not currently being accepted.

---

## License

This project is private and proprietary. All rights reserved.

---

## Credits

See [CREDITS.md](CREDITS.md) for acknowledgments and attributions.

---

## Support

For questions or issues, please refer to the documentation files or contact the project maintainer.

**Project Status**: Active Development | Production Ready for Data Collection Phase

**Version**: 2.0
**Last Updated**: 2025-11-10
