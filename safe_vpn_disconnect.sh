#!/bin/bash
# Safe VPN Disconnection Script

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

VPN_INTERFACE="wg0"

echo -e "${CYAN}[VPN]${NC} Disconnecting..."

if ip link show "$VPN_INTERFACE" &> /dev/null; then
    wg-quick down "$VPN_INTERFACE" &> /dev/null || true
    sleep 2

    if ! ip link show "$VPN_INTERFACE" &> /dev/null; then
        echo -e "${GREEN}✓ VPN disconnected${NC}"

        # Test connectivity restored
        if ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
            echo -e "${GREEN}✓ Internet restored${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to disconnect${NC}"
    fi
else
    echo -e "${YELLOW}VPN not connected${NC}"
fi
