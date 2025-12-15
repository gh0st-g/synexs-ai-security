#!/bin/bash
# Secure SYNEXS Ports - Block sensitive services from external access

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Securing SYNEXS Ports ===${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}ERROR: Please run as root${NC}"
    exit 1
fi

# Enable UFW if not enabled
if ! ufw status | grep -q "Status: active"; then
    echo -e "${GREEN}[1/5]${NC} Enabling UFW firewall..."

    # Allow SSH first to prevent lockout
    ufw allow 22/tcp comment 'SSH'

    # Enable firewall
    ufw --force enable
else
    echo -e "${GREEN}[1/5]${NC} UFW already enabled"
fi

# Allow required public ports
echo -e "${GREEN}[2/5]${NC} Allowing public services..."
ufw allow 22/tcp comment 'SSH' 2>/dev/null || true
ufw allow 80/tcp comment 'HTTP' 2>/dev/null || true
ufw allow 443/tcp comment 'HTTPS' 2>/dev/null || true
ufw allow 2222/tcp comment 'Honeypot SSH' 2>/dev/null || true

# Block sensitive ports from external access (allow only localhost)
echo -e "${GREEN}[3/5]${NC} Blocking sensitive services from external access..."

# Health API - only localhost
ufw deny 8765/tcp comment 'Health API - localhost only' 2>/dev/null || true

# Redis - only localhost
ufw deny 6379/tcp comment 'Redis - localhost only' 2>/dev/null || true

# Listener - only localhost
ufw deny 5555/tcp comment 'Listener - localhost only' 2>/dev/null || true

# PostgreSQL - only localhost or Docker network
ufw deny from any to any port 5432 comment 'PostgreSQL - restricted' 2>/dev/null || true

# Dashboard - localhost only (access via nginx proxy if needed)
ufw deny 5000/tcp comment 'Dashboard - localhost only' 2>/dev/null || true

# Honeypot internal port
ufw deny 8080/tcp comment 'Honeypot backend - localhost only' 2>/dev/null || true

# Allow Docker networks (for inter-container communication)
echo -e "${GREEN}[4/5]${NC} Allowing Docker networks..."
ufw allow from 172.16.0.0/12 comment 'Docker networks' 2>/dev/null || true
ufw allow from 10.0.0.0/8 comment 'Private networks' 2>/dev/null || true

# Reload firewall
echo -e "${GREEN}[5/5]${NC} Reloading firewall..."
ufw reload

echo ""
echo -e "${GREEN}=== Firewall Configuration Complete ===${NC}"
echo ""
echo "Current status:"
ufw status verbose

echo ""
echo -e "${YELLOW}Publicly accessible ports:${NC}"
echo "  - 22   (SSH)"
echo "  - 80   (HTTP)"
echo "  - 443  (HTTPS)"
echo "  - 2222 (Honeypot SSH)"
echo ""
echo -e "${GREEN}Protected ports (localhost only):${NC}"
echo "  - 5000 (Dashboard)"
echo "  - 5432 (PostgreSQL)"
echo "  - 5555 (Listener)"
echo "  - 6379 (Redis)"
echo "  - 8080 (Honeypot backend)"
echo "  - 8765 (Health API)"
echo ""
echo -e "${GREEN}✓ Your VPS is now more secure!${NC}"
