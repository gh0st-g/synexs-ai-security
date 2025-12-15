#!/bin/bash
# Port Scanner and Network Analysis Script
# Checks what ports are actually open on this VPS

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   VPS Port & Network Analysis Tool        ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Note: Running as non-root, some info may be limited${NC}"
    echo ""
fi

# 1. Get listening ports with processes
echo -e "${GREEN}[1/5] Listening Ports (with process names)${NC}"
echo -e "${YELLOW}Port    Proto   Process${NC}"
echo "----------------------------------------"
if command -v ss &> /dev/null; then
    ss -tulnp | grep LISTEN | awk '{print $5, $1, $7}' | sed 's/.*://g' | sort -n | column -t
elif command -v netstat &> /dev/null; then
    netstat -tulnp | grep LISTEN | awk '{print $4, $1, $7}' | sed 's/.*://g' | sort -n | column -t
else
    echo "Neither ss nor netstat available"
fi
echo ""

# 2. Get listening services detail
echo -e "${GREEN}[2/5] Detailed Service Information${NC}"
echo "----------------------------------------"
if command -v ss &> /dev/null; then
    ss -tlnp | grep LISTEN
elif command -v netstat &> /dev/null; then
    netstat -tlnp | grep LISTEN
fi
echo ""

# 3. Check firewall rules
echo -e "${GREEN}[3/5] Firewall Status${NC}"
echo "----------------------------------------"
if command -v ufw &> /dev/null; then
    echo "UFW Status:"
    ufw status verbose 2>/dev/null || echo "UFW not configured"
elif command -v iptables &> /dev/null; then
    echo "iptables rules (INPUT chain):"
    iptables -L INPUT -n -v 2>/dev/null || echo "Cannot read iptables (need root)"
else
    echo "No firewall tools found"
fi
echo ""

# 4. Check Docker exposed ports
echo -e "${GREEN}[4/5] Docker Container Ports${NC}"
echo "----------------------------------------"
if command -v docker &> /dev/null; then
    docker ps --format "table {{.Names}}\t{{.Ports}}" 2>/dev/null || echo "No running containers or no permission"
else
    echo "Docker not installed"
fi
echo ""

# 5. External port scan (from perspective of outside world)
echo -e "${GREEN}[5/5] External Port Scan (Installing nmap if needed)${NC}"
echo "----------------------------------------"

# Install nmap if not present
if ! command -v nmap &> /dev/null; then
    echo "Installing nmap..."
    apt-get update -qq && apt-get install -y nmap -qq
fi

# Get public IP
PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "157.245.3.180")
echo "Scanning external ports on: $PUBLIC_IP"
echo ""

# Common ports scan
nmap -Pn -p 22,80,443,2222,5555,8080,8765,3000,5000,6379,5432 $PUBLIC_IP 2>/dev/null | grep -E "PORT|open|closed|filtered" || echo "Scan failed"

echo ""
echo -e "${GREEN}Full port scan (top 1000 ports):${NC}"
nmap -Pn --top-ports 1000 $PUBLIC_IP 2>/dev/null | grep -E "PORT|open" || echo "No open ports found"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Summary & Recommendations                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Expected SYNEXS Ports:${NC}"
echo "  - Port 22    (SSH) - Should be open"
echo "  - Port 2222  (Honeypot SSH) - Should be open"
echo "  - Port 5555  (Listener) - Should be FILTERED/CLOSED from outside"
echo "  - Port 8765  (Health API) - Should be FILTERED/CLOSED from outside"
echo "  - Port 6379  (Redis) - Should be FILTERED/CLOSED from outside"
echo "  - Port 5432  (PostgreSQL) - Should be FILTERED/CLOSED from outside"
echo ""
echo -e "${GREEN}✓ Scan complete!${NC}"
echo ""
