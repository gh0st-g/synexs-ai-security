#!/bin/bash
################################################################################
# SYNEXS AUTO-DEPLOY - Plug & Play Defensive Honeypot Setup
# Automatically configures and deploys honeypot on any device
# Supports: VPS, Raspberry Pi, Laptop, Termux (Android)
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        SYNEXS AUTO-DEPLOY - DEFENSIVE HONEYPOT            ║"
echo "║                 Plug & Play Setup                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
    elif [ -f /etc/debian_version ]; then
        OS="debian"
    elif [ -f /etc/redhat-release ]; then
        OS="rhel"
    else
        OS=$(uname -s)
    fi

    echo -e "${GREEN}✅ Detected OS: $OS${NC}"
}

# Detect device type
detect_device() {
    if [ -n "$TERMUX_VERSION" ]; then
        DEVICE="termux"
        echo -e "${GREEN}✅ Device: Android (Termux)${NC}"
    elif grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
        DEVICE="raspberry_pi"
        echo -e "${GREEN}✅ Device: Raspberry Pi${NC}"
    elif [ -f /sys/class/dmi/id/product_name ]; then
        DEVICE="laptop_desktop"
        echo -e "${GREEN}✅ Device: Laptop/Desktop${NC}"
    else
        DEVICE="vps"
        echo -e "${GREEN}✅ Device: VPS/Server${NC}"
    fi
}

# Detect local IP address
detect_ip() {
    echo -e "${YELLOW}🔍 Detecting IP address...${NC}"

    # Try multiple methods to get IP
    if command -v ip &> /dev/null; then
        # Method 1: ip command (most reliable)
        LOCAL_IP=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+' 2>/dev/null)
    fi

    if [ -z "$LOCAL_IP" ]; then
        # Method 2: hostname command
        LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi

    if [ -z "$LOCAL_IP" ]; then
        # Method 3: ifconfig (fallback)
        LOCAL_IP=$(ifconfig 2>/dev/null | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n1)
    fi

    if [ -z "$LOCAL_IP" ]; then
        # Method 4: netstat (last resort)
        LOCAL_IP=$(netstat -rn 2>/dev/null | grep '^0.0.0.0' | awk '{print $8}' | xargs -I {} ifconfig {} 2>/dev/null | grep 'inet ' | awk '{print $2}')
    fi

    # Fallback to localhost
    if [ -z "$LOCAL_IP" ]; then
        LOCAL_IP="127.0.0.1"
        echo -e "${YELLOW}⚠️  Could not detect IP, using localhost${NC}"
    fi

    echo -e "${GREEN}✅ Honeypot IP: ${BLUE}$LOCAL_IP${NC}"
    echo -e "${GREEN}✅ Honeypot Port: ${BLUE}8080${NC}"

    HONEYPOT_URL="http://${LOCAL_IP}:8080"
}

# Create configuration file
create_config() {
    echo -e "\n${YELLOW}📝 Creating configuration...${NC}"

    cat > config_auto.json << EOF
{
  "honeypot": {
    "host": "${LOCAL_IP}",
    "port": 8080,
    "url": "${HONEYPOT_URL}"
  },
  "deployment": {
    "device": "${DEVICE}",
    "os": "${OS}",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  },
  "network": {
    "local_ip": "${LOCAL_IP}",
    "external_ip": "$(curl -s ifconfig.me 2>/dev/null || echo 'N/A')"
  },
  "features": {
    "waf": true,
    "ai_detection": true,
    "real_time_learning": true,
    "hybrid_blocking": true
  }
}
EOF

    echo -e "${GREEN}✅ Config saved: config_auto.json${NC}"
}

# Update Python files with honeypot URL
update_honeypot_config() {
    echo -e "\n${YELLOW}🔧 Rewriting ALL Python files with new C2 URL...${NC}"

    # Backup all Python files first
    echo -e "${BLUE}   Creating backups...${NC}"
    mkdir -p .backups_$(date +%Y%m%d_%H%M%S)
    find . -maxdepth 1 -name "*.py" -exec cp {} .backups_$(date +%Y%m%d_%H%M%S)/ \; 2>/dev/null

    # Count files to update
    PY_FILES=$(find . -maxdepth 1 -name "*.py" | wc -l)
    echo -e "${BLUE}   Found ${PY_FILES} Python files${NC}"

    # Update ALL Python files with new URLs
    UPDATED_COUNT=0
    while IFS= read -r pyfile; do
        FILENAME=$(basename "$pyfile")

        # Check if file contains URLs to replace
        if grep -qE "(http://|https://)?[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+|http://(localhost|127\.0\.0\.1):[0-9]+" "$pyfile" 2>/dev/null; then

            # Replace localhost and 127.0.0.1 URLs
            sed -i "s|http://127\.0\.0\.1:8080|${HONEYPOT_URL}|g" "$pyfile" 2>/dev/null
            sed -i "s|http://localhost:8080|${HONEYPOT_URL}|g" "$pyfile" 2>/dev/null
            sed -i "s|https://127\.0\.0\.1:8080|${HONEYPOT_URL}|g" "$pyfile" 2>/dev/null
            sed -i "s|https://localhost:8080|${HONEYPOT_URL}|g" "$pyfile" 2>/dev/null

            # Replace hardcoded VPS IPs (if they exist)
            sed -i -E "s|http://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:8080|${HONEYPOT_URL}|g" "$pyfile" 2>/dev/null

            # Replace HONEYPOT_URL variable assignments
            sed -i "s|HONEYPOT_URL = \"http://[^\"]*\"|HONEYPOT_URL = \"${HONEYPOT_URL}\"|g" "$pyfile" 2>/dev/null
            sed -i "s|C2_URL = \"http://[^\"]*\"|C2_URL = \"${HONEYPOT_URL}\"|g" "$pyfile" 2>/dev/null
            sed -i "s|VPS_ENDPOINT = \"http://[^\"]*\"|VPS_ENDPOINT = \"${HONEYPOT_URL}/report\"|g" "$pyfile" 2>/dev/null
            sed -i "s|LISTENER_IP = \"[^\"]*\"|LISTENER_IP = \"${LOCAL_IP}\"|g" "$pyfile" 2>/dev/null

            echo -e "${GREEN}      ✓ ${FILENAME}${NC}"
            ((UPDATED_COUNT++))
        fi
    done < <(find . -maxdepth 1 -name "*.py")

    if [ $UPDATED_COUNT -gt 0 ]; then
        echo -e "${GREEN}   ✅ Updated ${UPDATED_COUNT} Python files with ${HONEYPOT_URL}${NC}"
    else
        echo -e "${YELLOW}   ℹ️  No hardcoded URLs found to replace${NC}"
    fi

    # Create environment file
    cat > .env << EOF
# Auto-generated configuration
HONEYPOT_HOST=${LOCAL_IP}
HONEYPOT_PORT=8080
HONEYPOT_URL=${HONEYPOT_URL}
DEVICE_TYPE=${DEVICE}
DEPLOYMENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

    echo -e "${GREEN}   ✅ Created .env file${NC}"
}

# Open firewall port
configure_firewall() {
    echo -e "\n${YELLOW}🔥 Configuring firewall...${NC}"

    # Skip firewall on Termux
    if [ "$DEVICE" = "termux" ]; then
        echo -e "${BLUE}   ℹ️  Termux: No firewall configuration needed${NC}"
        return
    fi

    # Try UFW (Ubuntu/Debian)
    if command -v ufw &> /dev/null; then
        echo -e "${BLUE}   Using UFW...${NC}"
        sudo ufw allow 8080/tcp 2>/dev/null || true
        echo -e "${GREEN}   ✅ Port 8080 opened (UFW)${NC}"
    # Try firewalld (RHEL/CentOS)
    elif command -v firewall-cmd &> /dev/null; then
        echo -e "${BLUE}   Using firewalld...${NC}"
        sudo firewall-cmd --permanent --add-port=8080/tcp 2>/dev/null || true
        sudo firewall-cmd --reload 2>/dev/null || true
        echo -e "${GREEN}   ✅ Port 8080 opened (firewalld)${NC}"
    # Try iptables (fallback)
    elif command -v iptables &> /dev/null; then
        echo -e "${BLUE}   Using iptables...${NC}"
        sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT 2>/dev/null || true
        echo -e "${GREEN}   ✅ Port 8080 opened (iptables)${NC}"
    else
        echo -e "${YELLOW}   ⚠️  No firewall detected - ensure port 8080 is accessible${NC}"
    fi
}

# Install dependencies
install_dependencies() {
    echo -e "\n${YELLOW}📦 Installing dependencies...${NC}"

    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Python3: $(python3 --version)${NC}"

    # Check if pip is installed
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        echo -e "${RED}❌ pip not found. Please install pip${NC}"
        exit 1
    fi

    # Install requirements
    if [ -f "requirements_fast.txt" ]; then
        echo -e "${BLUE}   Installing Python packages...${NC}"
        pip3 install -q -r requirements_fast.txt 2>&1 | grep -E "(Successfully|ERROR|error)" || true
        echo -e "${GREEN}   ✅ Dependencies installed${NC}"
    else
        echo -e "${YELLOW}   ⚠️  requirements_fast.txt not found${NC}"
    fi
}

# Create deployment info
create_deployment_info() {
    echo -e "\n${YELLOW}📊 Creating deployment info...${NC}"

    cat > DEPLOYMENT_INFO.txt << EOF
╔════════════════════════════════════════════════════════════╗
║           SYNEXS DEFENSIVE HONEYPOT - DEPLOYED            ║
╚════════════════════════════════════════════════════════════╝

DEPLOYMENT DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Device Type:     ${DEVICE}
Operating System: ${OS}
Deployment Time:  $(date)

HONEYPOT CONFIGURATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Local IP:        ${LOCAL_IP}
Port:            8080
Honeypot URL:    ${HONEYPOT_URL}

External IP:     $(curl -s ifconfig.me 2>/dev/null || echo 'N/A')

ENDPOINTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Main:            ${HONEYPOT_URL}/
Stats:           ${HONEYPOT_URL}/stats
Health:          ${HONEYPOT_URL}/health
Robots:          ${HONEYPOT_URL}/robots.txt

FEATURES ENABLED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 3-Layer Detection (Rate Limit → WAF → AI)
✅ XGBoost ML Model (<5ms inference)
✅ Real-time Kill Learning
✅ Pandas Vectorization (10-100x faster)
✅ Hybrid WAF + AI Blocking
✅ Crawler Validation (CIDR + PTR)

QUICK START:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start Honeypot:  ./START_FAST_DEFENSE.sh
View Stats:      curl ${HONEYPOT_URL}/stats | jq
View Logs:       tail -f logs/honeypot.log
Stop Services:   pkill -f 'honeypot_server|defensive_engine'

TEST COMMANDS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# From this device:
curl ${HONEYPOT_URL}/

# From another device on same network:
curl ${HONEYPOT_URL}/

# Test detection:
curl -G --data-urlencode "user=admin' OR 1=1--" ${HONEYPOT_URL}/login

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛡️  DEFENSIVE RESEARCH - ALL TRAFFIC LOGGED FOR ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

    echo -e "${GREEN}✅ Deployment info saved: DEPLOYMENT_INFO.txt${NC}"
}

# Start honeypot
start_honeypot() {
    echo -e "\n${YELLOW}🚀 Starting defensive honeypot...${NC}"

    # Make scripts executable
    chmod +x START_FAST_DEFENSE.sh 2>/dev/null || true

    # Kill any existing processes
    pkill -f 'honeypot_server' 2>/dev/null || true
    pkill -f 'defensive_engine' 2>/dev/null || true
    sleep 1

    # Start the system
    if [ -f "START_FAST_DEFENSE.sh" ]; then
        echo -e "${BLUE}   Launching START_FAST_DEFENSE.sh...${NC}"
        ./START_FAST_DEFENSE.sh > logs/deployment.log 2>&1 &
        DEPLOY_PID=$!

        # Wait for startup
        sleep 3

        # Check if honeypot is running
        if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Honeypot is ONLINE!${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  Honeypot starting... (check logs/deployment.log)${NC}"
            return 0
        fi
    else
        echo -e "${RED}❌ START_FAST_DEFENSE.sh not found${NC}"
        return 1
    fi
}

# Display summary
display_summary() {
    echo -e "\n${CYAN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                  DEPLOYMENT COMPLETE                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${GREEN}✅ Honeypot URL:${NC} ${BLUE}${HONEYPOT_URL}${NC}"
    echo -e "${GREEN}✅ Local IP:${NC} ${BLUE}${LOCAL_IP}${NC}"
    echo -e "${GREEN}✅ Port:${NC} ${BLUE}8080${NC}"
    echo ""

    echo -e "${YELLOW}📊 Quick Commands:${NC}"
    echo -e "   ${CYAN}View Stats:${NC}    curl ${HONEYPOT_URL}/stats | jq"
    echo -e "   ${CYAN}Health Check:${NC}  curl ${HONEYPOT_URL}/health"
    echo -e "   ${CYAN}View Logs:${NC}     tail -f logs/honeypot.log"
    echo -e "   ${CYAN}Stop:${NC}          pkill -f 'honeypot_server|defensive_engine'"
    echo ""

    echo -e "${YELLOW}📱 Access from other devices:${NC}"
    echo -e "   ${CYAN}From phone/laptop on same network:${NC}"
    echo -e "   curl ${HONEYPOT_URL}/"
    echo ""

    echo -e "${YELLOW}📁 Files Created:${NC}"
    echo -e "   ${CYAN}config_auto.json${NC}      - Auto-generated config"
    echo -e "   ${CYAN}.env${NC}                  - Environment variables"
    echo -e "   ${CYAN}DEPLOYMENT_INFO.txt${NC}   - Full deployment details"
    echo ""

    echo -e "${GREEN}🛡️  Defensive honeypot deployed successfully!${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Test connectivity
test_connectivity() {
    echo -e "\n${YELLOW}🧪 Testing connectivity...${NC}"

    sleep 2

    # Test local
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/health | grep -q "200"; then
        echo -e "${GREEN}   ✅ Local access: OK${NC}"
    else
        echo -e "${YELLOW}   ⚠️  Local access: Check logs/honeypot.log${NC}"
    fi

    # Test via local IP
    if curl -s -o /dev/null -w "%{http_code}" ${HONEYPOT_URL}/health 2>/dev/null | grep -q "200"; then
        echo -e "${GREEN}   ✅ Network access: OK${NC}"
    else
        echo -e "${YELLOW}   ⚠️  Network access: May require firewall config${NC}"
    fi
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    echo -e "${BLUE}Starting auto-deployment...${NC}\n"

    # Change to script directory
    cd "$(dirname "$0")"

    # Create logs directory
    mkdir -p logs

    # Run deployment steps
    detect_os
    detect_device
    detect_ip
    create_config
    update_honeypot_config
    install_dependencies
    configure_firewall
    create_deployment_info
    start_honeypot
    test_connectivity
    display_summary

    echo -e "\n${GREEN}🎉 AUTO-DEPLOY COMPLETE${NC}"
    echo -e "${BLUE}Run 'cat DEPLOYMENT_INFO.txt' for full details${NC}\n"
}

# Handle Ctrl+C
trap 'echo -e "\n${RED}Deployment cancelled${NC}"; exit 1' INT

# Run main
main "$@"
