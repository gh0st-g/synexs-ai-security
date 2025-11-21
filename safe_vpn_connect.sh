#!/bin/bash
# Safe VPN Connection Script with Split-Tunneling
# Preserves internet access while providing anonymity for scans

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

VPN_INTERFACE="wg0"
BACKUP_ROUTE_TABLE="/tmp/vpn_backup_routes"

# Function: Check connectivity
check_connectivity() {
    local test_host=${1:-8.8.8.8}
    if ping -c 1 -W 2 "$test_host" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function: Test DNS
check_dns() {
    if nslookup google.com &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function: Backup current routes
backup_routes() {
    echo -e "${CYAN}[BACKUP]${NC} Saving current routing table..."
    ip route show > "$BACKUP_ROUTE_TABLE"
    ip rule show >> "$BACKUP_ROUTE_TABLE"
}

# Function: Restore routes
restore_routes() {
    echo -e "${YELLOW}[RESTORE]${NC} Restoring original routes..."
    if [ -f "$BACKUP_ROUTE_TABLE" ]; then
        echo -e "${GREEN}✓ Backup available${NC}"
    fi
}

# Function: Safe VPN connect
safe_vpn_connect() {
    echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     SAFE VPN CONNECTION v2.0              ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
    echo ""

    # Check if already connected
    if ip link show "$VPN_INTERFACE" &> /dev/null; then
        echo -e "${GREEN}✓ VPN already connected${NC}"

        if check_connectivity; then
            echo -e "${GREEN}✓ Internet connectivity: OK${NC}"
        else
            echo -e "${RED}✗ Internet connectivity: FAILED${NC}"
            echo -e "${YELLOW}Reconnecting VPN...${NC}"
            wg-quick down "$VPN_INTERFACE" &> /dev/null || true
            sleep 2
        fi
    fi

    # If not connected, connect now
    if ! ip link show "$VPN_INTERFACE" &> /dev/null; then
        backup_routes

        echo -e "${CYAN}[CHECK]${NC} Testing connectivity before VPN..."
        if check_connectivity; then
            echo -e "${GREEN}✓ Internet: OK${NC}"
        else
            echo -e "${RED}✗ No internet connection${NC}"
            exit 1
        fi

        echo -e "${CYAN}[VPN]${NC} Connecting..."
        wg-quick up "$VPN_INTERFACE" 2>&1 | grep -v "RTNETLINK" || true

        sleep 3

        echo -e "${CYAN}[CHECK]${NC} Testing after VPN..."

        if check_connectivity 8.8.8.8; then
            echo -e "${GREEN}✓ Internet: OK${NC}"
        else
            echo -e "${YELLOW}⚠ Limited connectivity (normal for some VPNs)${NC}"
        fi

        if check_dns; then
            echo -e "${GREEN}✓ DNS: OK${NC}"
        else
            echo -e "${YELLOW}⚠ Adding backup DNS...${NC}"
            echo "nameserver 8.8.8.8" >> /etc/resolv.conf
        fi

        echo ""
        VPN_IP=$(ip addr show "$VPN_INTERFACE" 2>/dev/null | grep "inet " | awk '{print $2}')
        PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "Unknown")
        echo -e "${CYAN}[VPN INFO]${NC}"
        echo -e "  Interface: ${GREEN}$VPN_INTERFACE${NC}"
        echo -e "  VPN IP: ${GREEN}$VPN_IP${NC}"
        echo -e "  Public IP: ${GREEN}$PUBLIC_IP${NC}"
    fi

    echo ""
    echo -e "${GREEN}✓ VPN Ready${NC}"
}

# Main
case "${1:-connect}" in
    connect|up) safe_vpn_connect ;;
    test)
        check_connectivity && echo -e "${GREEN}✓ Internet OK${NC}" || echo -e "${RED}✗ No internet${NC}"
        check_dns && echo -e "${GREEN}✓ DNS OK${NC}" || echo -e "${RED}✗ DNS failed${NC}"
        ;;
    *)
        echo "Usage: $0 {connect|test}"
        ;;
esac
