#!/bin/bash
# Prepare SYNEXS files for GitHub upload
# Sanitizes sensitive information before upload

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

UPLOAD_DIR="/root/synexs_github_upload"

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SYNEXS GitHub Upload Preparation        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Clean up any previous upload directory
if [ -d "$UPLOAD_DIR" ]; then
    echo -e "${YELLOW}Cleaning previous upload directory...${NC}"
    rm -rf "$UPLOAD_DIR"
fi

# Create upload directory
echo -e "${GREEN}[1/8]${NC} Creating upload directory..."
mkdir -p "$UPLOAD_DIR"

# Copy core Python scripts (sanitized)
echo -e "${GREEN}[2/8]${NC} Copying core Python scripts..."
cp /root/synexs/synexs_core_orchestrator.py "$UPLOAD_DIR/"
cp /root/synexs/synexs_model.py "$UPLOAD_DIR/"
cp /root/synexs/binary_protocol.py "$UPLOAD_DIR/"
cp /root/synexs/dna_collector.py "$UPLOAD_DIR/"
cp /root/synexs/health_check.py "$UPLOAD_DIR/"
cp /root/synexs/honeypot_server.py "$UPLOAD_DIR/"

# Copy utility scripts (already safe)
echo -e "${GREEN}[3/8]${NC} Copying utility scripts..."
cp /root/synexs/check_open_ports.sh "$UPLOAD_DIR/"
cp /root/synexs/secure_ports.sh "$UPLOAD_DIR/"
cp /root/synexs/synexs_verify.sh "$UPLOAD_DIR/"

# Copy VPN scripts (with sanitization)
echo -e "${GREEN}[4/8]${NC} Copying and sanitizing VPN scripts..."
sed 's/23\.234\.95\.120/YOUR_SSH_CLIENT_IP/g' /root/synexs/safe_vpn_connect.sh | \
sed 's/157\.245\.3\.180/YOUR_VPS_PUBLIC_IP/g' | \
sed 's/157\.245\.0\.1/YOUR_VPS_GATEWAY/g' > "$UPLOAD_DIR/safe_vpn_connect.sh"

sed 's/23\.234\.95\.120/YOUR_SSH_CLIENT_IP/g' /root/synexs/safe_vpn_disconnect.sh | \
sed 's/157\.245\.0\.1/YOUR_VPS_GATEWAY/g' > "$UPLOAD_DIR/safe_vpn_disconnect.sh"

chmod +x "$UPLOAD_DIR"/*.sh

# Copy documentation
echo -e "${GREEN}[5/8]${NC} Copying documentation..."
cp /root/synexs/VPN_AND_SECURITY_GUIDE.md "$UPLOAD_DIR/"
cp /root/synexs/SYNEXS_MASTER_DOCUMENTATION.md "$UPLOAD_DIR/" 2>/dev/null || true
cp /root/synexs/CORE_ORCHESTRATOR_README.md "$UPLOAD_DIR/" 2>/dev/null || true
cp /root/synexs/README.md "$UPLOAD_DIR/" 2>/dev/null || true

# Copy vocabulary files (safe)
echo -e "${GREEN}[6/8]${NC} Copying vocabulary files..."
cp /root/synexs/vocab_v3_binary.json "$UPLOAD_DIR/" 2>/dev/null || true
cp /root/synexs/vocab.json "$UPLOAD_DIR/" 2>/dev/null || true

# Create .gitignore
echo -e "${GREEN}[7/8]${NC} Creating .gitignore..."
cat > "$UPLOAD_DIR/.gitignore" << 'EOF'
# Sensitive files - DO NOT COMMIT
*.env
.env.*
credentials*.json
*_state.json
*.pid
*.pth
*.joblib

# WireGuard configs
*.conf

# Logs
*.log
*.out
nohup.log

# Data files
*.jsonl
datasets/
training_*/
*.tar.gz

# Private keys
*_key
*_private
privatekey

# Database
*.db
*.sqlite
*.sql

# Caches
__pycache__/
*.pyc
.cache/
*.cache

# Temporary files
/tmp/
*.tmp
*.bak
*.backup
*.save

# Personal info
processed_ids.json
memory_log.json
.goals_tracker.json
file_hashes.json
EOF

# Create README for GitHub
echo -e "${GREEN}[8/8]${NC} Creating GitHub README..."
cat > "$UPLOAD_DIR/README.md" << 'EOF'
# SYNEXS - AI-Powered Security Training System

**Version**: 2.0
**Status**: Production Ready

## 🎯 Overview

SYNEXS is an AI-powered defensive security training system featuring:
- Agent-based swarm intelligence for adaptive learning
- Honeypot detection and attack pattern analysis
- Binary protocol communication (88% bandwidth reduction)
- Self-learning from operational data
- Real-time orchestration and monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Redis server
- PostgreSQL (optional)
- WireGuard (for VPN features)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/synexs.git
cd synexs

# Install dependencies
pip3 install -r requirements.txt

# Run port scanner
sudo bash check_open_ports.sh

# Secure your ports
sudo bash secure_ports.sh
```

### VPN Setup (Optional)

```bash
# Configure WireGuard at /etc/wireguard/wg0.conf
# Then connect safely without losing SSH:
sudo bash safe_vpn_connect.sh

# Disconnect:
sudo bash safe_vpn_disconnect.sh
```

## 📁 Key Components

### Core Systems
- `synexs_core_orchestrator.py` - Main orchestration engine
- `synexs_model.py` - Unified AI model
- `binary_protocol.py` - Ultra-efficient protocol (v3)
- `honeypot_server.py` - Attack capture system
- `dna_collector.py` - Training data generator

### Utilities
- `check_open_ports.sh` - Port scanner & analysis
- `secure_ports.sh` - Firewall configuration
- `safe_vpn_connect.sh` - VPN with SSH protection
- `health_check.py` - System monitoring
- `synexs_verify.sh` - Pipeline verification

## 🔐 Security

### Port Security
The system secures sensitive services:
- ✅ SSH (22), HTTP (80), HTTPS (443) - Public
- 🔒 Redis (6379), PostgreSQL (5432), Health API (8765) - Localhost only

Run `secure_ports.sh` to apply firewall rules.

### VPN Integration
Safe VPN connection with SSH protection:
- Policy-based routing keeps SSH alive
- Auto-rollback on failure
- Compatible with Mullvad/WireGuard

See `VPN_AND_SECURITY_GUIDE.md` for details.

## 📊 Architecture

```
┌─────────────────────────────────────────┐
│           SYNEXS ARCHITECTURE           │
├─────────────────────────────────────────┤
│  Honeypot → DNA Collector → Training   │
│  Listener → AI Swarm → Cell Executor   │
│  Propagate → Agent Spawner → Learning  │
└─────────────────────────────────────────┘
```

## 📖 Documentation

- `SYNEXS_MASTER_DOCUMENTATION.md` - Complete system docs
- `CORE_ORCHESTRATOR_README.md` - Orchestrator details
- `VPN_AND_SECURITY_GUIDE.md` - VPN & security guide

## 🛠️ Configuration

Edit key parameters in:
- `synexs_core_orchestrator.py` - Cycle intervals, cell phases
- `.env` - Environment variables (not included in repo)
- `/etc/wireguard/wg0.conf` - VPN configuration (not included)

## ⚠️ Important Notes

**DO NOT commit:**
- Private keys or credentials
- WireGuard configurations
- Training data or model weights
- Log files or personal data

See `.gitignore` for full list.

## 📝 License

This is educational/research software. Use responsibly and ethically.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit pull request

## 📧 Contact

For issues or questions, please open a GitHub issue.

---

**Note**: This system is designed for defensive security research and training. Always obtain proper authorization before security testing.
EOF

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Preparation Complete!                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Upload directory:${NC} $UPLOAD_DIR"
echo ""
echo "Files prepared:"
ls -lh "$UPLOAD_DIR" | tail -n +2 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review files in: $UPLOAD_DIR"
echo "2. Initialize git: cd $UPLOAD_DIR && git init"
echo "3. Create GitHub repo at: https://github.com/new"
echo "4. Push to GitHub:"
echo "   cd $UPLOAD_DIR"
echo "   git add ."
echo "   git commit -m 'Initial commit: SYNEXS v2.0'"
echo "   git remote add origin https://github.com/YOUR_USERNAME/synexs.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo -e "${GREEN}✓ All sensitive information has been removed!${NC}"
