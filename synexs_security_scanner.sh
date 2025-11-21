#!/bin/bash
# SYNEXS Security Scanner - VPN + NMAP Security Testing Tool
# Author: gh0st-g
# Purpose: Comprehensive security scanning for YOUR OWN servers/lab ONLY
# ⚠️  ONLY USE ON SYSTEMS YOU OWN OR HAVE PERMISSION TO SCAN

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
VPN_INTERFACE="wg0"
OUTPUT_DIR="./scan_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Banner
cat << 'EOF'
╔═══════════════════════════════════════════════════════╗
║       SYNEXS SECURITY SCANNER v2.0                    ║
║       VPN + NMAP Security Testing Suite               ║
╚═══════════════════════════════════════════════════════╝
EOF

echo ""
echo -e "${RED}⚠️  WARNING: ONLY USE ON YOUR OWN SYSTEMS ⚠️${NC}"
echo -e "${YELLOW}Unauthorized scanning is ILLEGAL${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (needed for some scans)${NC}"
    exit 1
fi

# Install nmap if needed
if ! command -v nmap &> /dev/null; then
    echo -e "${YELLOW}Installing nmap...${NC}"
    apt-get update -qq && apt-get install -y nmap -qq
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function: Connect to VPN (uses safe script)
connect_vpn() {
    echo -e "${CYAN}[VPN]${NC} Checking VPN connection..."
    if ip link show "$VPN_INTERFACE" &> /dev/null; then
        echo -e "${GREEN}✓ VPN already connected${NC}"

        # Verify connectivity
        if ! ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
            echo -e "${YELLOW}⚠ No internet through VPN, reconnecting...${NC}"
            /root/synexs/safe_vpn_disconnect.sh
            /root/synexs/safe_vpn_connect.sh
        fi
        return 0
    fi

    echo -e "${YELLOW}Connecting to VPN safely...${NC}"
    if [ -f "/root/synexs/safe_vpn_connect.sh" ]; then
        /root/synexs/safe_vpn_connect.sh
    elif [ -f "/etc/wireguard/${VPN_INTERFACE}.conf" ]; then
        echo -e "${YELLOW}Using fallback connection method...${NC}"
        wg-quick up "$VPN_INTERFACE" 2>&1 | grep -v "RTNETLINK" || true
        sleep 2
        echo -e "${GREEN}✓ VPN connected${NC}"
    else
        echo -e "${YELLOW}! VPN not available, continuing without VPN${NC}"
    fi

    # Post-connection test
    echo -e "${CYAN}Testing connectivity...${NC}"
    if ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
        echo -e "${GREEN}✓ Internet access: OK${NC}"
    else
        echo -e "${YELLOW}⚠ Limited internet (VPN may block general traffic)${NC}"
        echo -e "${CYAN}You can still scan specific targets${NC}"
    fi
}

# Function: Disconnect VPN (uses safe script)
disconnect_vpn() {
    if ip link show "$VPN_INTERFACE" &> /dev/null; then
        echo -e "${CYAN}[VPN]${NC} Disconnecting VPN..."
        if [ -f "/root/synexs/safe_vpn_disconnect.sh" ]; then
            /root/synexs/safe_vpn_disconnect.sh
        else
            wg-quick down "$VPN_INTERFACE" &> /dev/null || true
            echo -e "${GREEN}✓ VPN disconnected${NC}"
        fi
    fi
}

# Function: Check target reachability
check_target() {
    local target=$1
    echo -e "${CYAN}[CHECK]${NC} Testing target reachability: $target"

    if ping -c 1 -W 3 "$target" &> /dev/null; then
        echo -e "${GREEN}✓ Target is reachable${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Target not responding to ping (may be firewalled)${NC}"
        echo -e "${CYAN}Proceeding with scan anyway...${NC}"
        return 0
    fi
}

# Function: Fast Recon Scan
fast_recon() {
    local target=$1
    local output="${OUTPUT_DIR}/fast_recon_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 1/8]${NC} Fast Reconnaissance (Top 1000 ports)"
    echo -e "${BLUE}Command:${NC} nmap -sS -sV -T4 --top-ports 1000 $target"

    check_target "$target"

    if nmap -sS -sV -T4 --top-ports 1000 "$target" -oN "${output}.txt" -oX "${output}.xml"; then
        echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
    else
        echo -e "${RED}✗ Scan failed, but partial results may be saved${NC}"
        echo -e "${YELLOW}Check: ${output}.txt${NC}"
    fi
}

# Function: Full Aggressive Scan
full_aggressive() {
    local target=$1
    local output="${OUTPUT_DIR}/full_aggressive_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 2/8]${NC} Full Aggressive Scan (All ports)"
    echo -e "${BLUE}Command:${NC} nmap -A -T4 -p- $target"
    echo -e "${YELLOW}⏱  This may take 10-30 minutes...${NC}"

    nmap -A -T4 -p- "$target" -oN "${output}.txt" -oX "${output}.xml"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
}

# Function: Version + OS Detection
version_os_scan() {
    local target=$1
    local output="${OUTPUT_DIR}/version_os_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 3/8]${NC} Version + OS Detection"
    echo -e "${BLUE}Command:${NC} nmap -sV -O -T4 $target"

    nmap -sV -O -T4 "$target" -oN "${output}.txt" -oX "${output}.xml"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
}

# Function: UDP Top Ports
udp_scan() {
    local target=$1
    local output="${OUTPUT_DIR}/udp_top200_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 4/8]${NC} UDP Top 200 Ports"
    echo -e "${BLUE}Command:${NC} nmap -sU --top-ports 200 $target"
    echo -e "${YELLOW}⏱  UDP scans are slow...${NC}"

    nmap -sU --top-ports 200 "$target" -oN "${output}.txt" -oX "${output}.xml"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
}

# Function: Vulnerability Scan
vuln_scan() {
    local target=$1
    local output="${OUTPUT_DIR}/vuln_scan_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 5/8]${NC} Vulnerability Scan"
    echo -e "${BLUE}Command:${NC} nmap --script vuln -p- $target"
    echo -e "${YELLOW}⏱  This may take a while...${NC}"

    nmap --script vuln -p- "$target" -oN "${output}.txt" -oX "${output}.xml"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
}

# Function: HTTP Deep Dive
http_scan() {
    local target=$1
    local output="${OUTPUT_DIR}/http_deep_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 6/8]${NC} HTTP Deep Dive"
    echo -e "${BLUE}Command:${NC} nmap --script http-* $target"

    nmap --script http-enum,http-methods,http-headers,http-robots.txt,http-title,http-auth \
        "$target" -oN "${output}.txt" -oX "${output}.xml"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
}

# Function: SSL/TLS Check
ssl_scan() {
    local target=$1
    local output="${OUTPUT_DIR}/ssl_check_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 7/8]${NC} SSL/TLS Security Check"
    echo -e "${BLUE}Command:${NC} nmap --script ssl-* -p 443 $target"

    nmap --script ssl-enum-ciphers,ssl-cert,ssl-heartbleed -p 443 "$target" \
        -oN "${output}.txt" -oX "${output}.xml"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
}

# Function: Nuclear Option (Everything)
nuclear_scan() {
    local target=$1
    local output="${OUTPUT_DIR}/nuclear_${TIMESTAMP}"

    echo -e "${CYAN}[SCAN 8/8]${NC} ${RED}NUCLEAR OPTION${NC} (Everything!)"
    echo -e "${BLUE}Command:${NC} nmap -A -T4 -p- --script vuln,exploit,http-*,ssl-* $target"
    echo -e "${YELLOW}⏱  This will take 30-60 minutes or more...${NC}"

    nmap -A -T4 -p- --script "vuln,exploit,http-*,ssl-*" "$target" \
        -oA "${output}"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.*"
}

# Function: Quick Website Test
quick_web_test() {
    local target=$1
    local output="${OUTPUT_DIR}/quick_web_${TIMESTAMP}"

    echo -e "${CYAN}[WEB TEST]${NC} Quick Website Security Check"
    echo -e "${BLUE}Command:${NC} nmap -p 80,443,8080,8443 --script http-*,ssl-* $target"

    nmap -p 80,443,8080,8443,3000,5000,8000 \
        --script http-enum,http-methods,http-headers,http-title,ssl-enum-ciphers \
        "$target" -oN "${output}.txt"

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
}

# Function: Honeypot Detection Test
honeypot_test() {
    local target=$1
    local output="${OUTPUT_DIR}/honeypot_test_${TIMESTAMP}"

    echo -e "${CYAN}[HONEYPOT]${NC} Testing Honeypot Detection on YOUR VPS"
    echo -e "${BLUE}Testing port 2222 (SSH Honeypot)${NC}"

    # Test SSH honeypot
    nmap -p 2222 -sV --script ssh-auth-methods,ssh-hostkey "$target" \
        -oN "${output}.txt"

    # Try to connect and trigger detection
    echo -e "${YELLOW}Attempting connection to trigger detection...${NC}"
    timeout 5 ssh -p 2222 -o StrictHostKeyChecking=no test@"$target" 2>&1 | head -5 >> "${output}.txt" || true

    echo -e "${GREEN}✓ Complete${NC} → ${output}.txt"
    echo -e "${CYAN}Check your honeypot logs for detection!${NC}"
}

# Main Menu
show_menu() {
    echo ""
    echo -e "${MAGENTA}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║         SCAN OPTIONS                      ║${NC}"
    echo -e "${MAGENTA}╚═══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Quick Scans:${NC}"
    echo "  1) Fast Recon (Top 1000 ports, 1-2 min)"
    echo "  2) Quick Web Test (HTTP/HTTPS only)"
    echo "  3) Honeypot Test (Test YOUR honeypot)"
    echo ""
    echo -e "${CYAN}Comprehensive Scans:${NC}"
    echo "  4) Version + OS Detection"
    echo "  5) UDP Top 200 Ports"
    echo "  6) Vulnerability Scan"
    echo "  7) HTTP Deep Dive"
    echo "  8) SSL/TLS Security Check"
    echo ""
    echo -e "${RED}Advanced:${NC}"
    echo "  9) Full Aggressive Scan (All ports)"
    echo " 10) Nuclear Option (EVERYTHING - very slow!)"
    echo ""
    echo -e "${CYAN}Utilities:${NC}"
    echo " 11) Connect/Disconnect VPN"
    echo " 12) View Previous Scan Results"
    echo "  0) Exit"
    echo ""
}

# View results function
view_results() {
    echo -e "${CYAN}Recent Scan Results:${NC}"
    ls -lht "$OUTPUT_DIR" | head -20 | awk '{print $9, "(" $5 ")"}'
    echo ""
    echo -e "${YELLOW}To view a file: cat ${OUTPUT_DIR}/filename${NC}"
}

# Main execution
main() {
    # Get target
    if [ -z "$1" ]; then
        echo -e "${YELLOW}Enter target (IP or domain):${NC} "
        read -r TARGET
    else
        TARGET=$1
    fi

    # Validate target
    if [ -z "$TARGET" ]; then
        echo -e "${RED}Error: No target specified${NC}"
        exit 1
    fi

    echo -e "${GREEN}Target: $TARGET${NC}"
    echo ""

    # VPN Option
    echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
    echo -e "${YELLOW}VPN Options:${NC}"
    echo "  y - Connect to VPN (anonymous scanning)"
    echo "  n - No VPN (direct connection - for your own servers)"
    echo "  t - Test VPN connectivity"
    echo ""
    echo -e "${YELLOW}Connect to VPN? (y/n/t):${NC} "
    read -r use_vpn

    if [[ "$use_vpn" =~ ^[Tt]$ ]]; then
        echo -e "${CYAN}Testing VPN connectivity...${NC}"
        /root/synexs/safe_vpn_connect.sh test 2>/dev/null || {
            ping -c 1 -W 2 8.8.8.8 &> /dev/null && echo -e "${GREEN}✓ Internet OK${NC}" || echo -e "${RED}✗ No internet${NC}"
        }
        echo ""
        echo -e "${YELLOW}Connect now? (y/n):${NC} "
        read -r use_vpn
    fi

    if [[ "$use_vpn" =~ ^[Yy]$ ]]; then
        connect_vpn
    else
        echo -e "${CYAN}Scanning without VPN (direct connection)${NC}"
    fi

    echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
    echo ""

    # Show menu
    while true; do
        show_menu
        echo -e "${YELLOW}Select scan type (0-12):${NC} "
        read -r choice

        case $choice in
            1) fast_recon "$TARGET" ;;
            2) quick_web_test "$TARGET" ;;
            3) honeypot_test "$TARGET" ;;
            4) version_os_scan "$TARGET" ;;
            5) udp_scan "$TARGET" ;;
            6) vuln_scan "$TARGET" ;;
            7) http_scan "$TARGET" ;;
            8) ssl_scan "$TARGET" ;;
            9) full_aggressive "$TARGET" ;;
            10) nuclear_scan "$TARGET" ;;
            11)
                if ip link show "$VPN_INTERFACE" &> /dev/null; then
                    disconnect_vpn
                else
                    connect_vpn
                fi
                ;;
            12) view_results ;;
            0)
                echo -e "${YELLOW}Disconnect VPN? (y/n):${NC} "
                read -r disc_vpn
                if [[ "$disc_vpn" =~ ^[Yy]$ ]]; then
                    disconnect_vpn
                fi
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *) echo -e "${RED}Invalid choice${NC}" ;;
        esac

        echo ""
        echo -e "${CYAN}Press Enter to continue...${NC}"
        read -r
    done
}

# Trap to disconnect VPN on exit
trap 'echo ""; echo "Cleaning up..."; disconnect_vpn' EXIT INT TERM

# Run main function
main "$@"
