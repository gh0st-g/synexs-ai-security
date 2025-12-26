# Bash Vulnerability Scanner

A lightweight, portable bash-based vulnerability scanner that requires NO Python dependencies!

## Features

✅ **Zero Python Dependencies** - Pure bash script using standard Unix tools
✅ **Fully Portable** - Works on any Linux/Unix system with bash
✅ **Fast & Lightweight** - Native performance
✅ **Same Capabilities** - All 11 vulnerability types from Python version
✅ **JSON Export** - Optional jq for formatted reports
✅ **Color-Coded Output** - Easy to read results
✅ **Verbose Mode** - Debug mode for troubleshooting

## Requirements

### Minimal (Required):
- `bash` (version 4.0+)
- `curl` - For HTTP requests
- `grep` - For pattern matching (usually pre-installed)
- `sed` - For text processing (usually pre-installed)

### Optional (Recommended):
- `jq` - For formatted JSON reports (install: `apt install jq` or `yum install jq`)

## Installation

```bash
# Download the script (already created)
cd /root/synexs

# Make executable
chmod +x vuln_scanner.sh

# Test it
./vuln_scanner.sh --help
```

## Usage

### Basic Scan (All Vulnerability Types)

```bash
./vuln_scanner.sh -u http://target.com
```

### Scan Specific Types

```bash
./vuln_scanner.sh -u http://target.com -t sqli,xss,rce
```

### Verbose Mode

```bash
./vuln_scanner.sh -u http://target.com -v
```

### Export to JSON

```bash
./vuln_scanner.sh -u http://target.com -o report.json
```

### Custom Timeout

```bash
./vuln_scanner.sh -u http://target.com --timeout 15
```

## Scan Types

Use `-t` to specify scan types (comma-separated):

- `sqli` - SQL Injection
- `xss` - Cross-Site Scripting
- `lfi` - Local File Inclusion
- `rce` - Remote Code Execution
- `ssrf` - Server-Side Request Forgery
- `xxe` - XML External Entity
- `ssti` - Server-Side Template Injection
- `nosqli` - NoSQL Injection
- `ldap` - LDAP Injection
- `graphql` - GraphQL Vulnerabilities
- `headers` - HTTP Header Injection
- `all` - Run all scans (default)

## Examples

### Scan Your Honeypot

```bash
./vuln_scanner.sh -u http://your-target.com -v -o honeypot_scan.json
```

### Quick SQL Injection Test

```bash
./vuln_scanner.sh -u http://target.com -t sqli
```

### Full Scan with Report

```bash
./vuln_scanner.sh -u https://webapp.example.com -o full_scan.json
```

### Focused Testing

```bash
# Test only injection attacks
./vuln_scanner.sh -u http://target.com -t sqli,xss,ssti,nosqli

# Test only file-related issues
./vuln_scanner.sh -u http://target.com -t lfi,xxe
```

## Output Example

```
╔══════════════════════════════════════════════════════════════╗
║     Web Application Vulnerability Scanner (Bash) v1.0       ║
║     For Authorized Security Testing Only                    ║
╚══════════════════════════════════════════════════════════════╝

Target: http://your-target.com
Scan Started: 2025-12-26 10:30:00

[1/11] Running SQL Injection scan...
  ✗ SQL Injection found: /login/?username=

[2/11] Running XSS scan...
  ✗ XSS found: /comments/?q=

[3/11] Running LFI scan...
  ✓ No LFI vulnerabilities found

...

╔══════════════════════════════════════════════════════════════╗
║                     SCAN SUMMARY                             ║
╚══════════════════════════════════════════════════════════════╝

Total Vulnerabilities Found: 15

  Critical: 4
  High:     6
  Medium:   5
  Low:      0

Scan Duration: 42s
```

## Python vs Bash Version Comparison

| Feature | Python Version | Bash Version |
|---------|---------------|--------------|
| **Dependencies** | Requires Python 3.x, requests, colorama, urllib3 | Only bash, curl, grep, sed |
| **Installation** | `pip install -r requirements.txt` | Already installed on most systems |
| **Portability** | Needs Python environment | Runs everywhere |
| **Performance** | Slightly slower startup | Faster startup |
| **Code Size** | ~300 lines + modules | ~850 lines single file |
| **Maintainability** | Better structure | Single file is simple |
| **JSON Output** | Native JSON support | Requires jq (optional) |
| **Error Handling** | More robust | Basic but effective |
| **Extensibility** | Modular design | All-in-one |

## When to Use Which Version

### Use Python Version When:
- You need complex data structures
- You want modular, extensible code
- You're integrating with other Python tools
- You need advanced HTTP features
- You prefer object-oriented design

### Use Bash Version When:
- You want zero dependencies
- You need maximum portability
- You're running on minimal systems
- You want fast deployment
- You prefer simple, readable scripts
- You're already in a shell environment

## Advanced Features

### Installing jq for Better Reports

```bash
# Debian/Ubuntu
apt install jq

# CentOS/RHEL
yum install jq

# macOS
brew install jq
```

### Reading JSON Reports

```bash
# View all critical vulnerabilities
cat report.json | jq '.vulnerabilities[] | select(.severity=="Critical")'

# Count vulnerabilities by type
cat report.json | jq '.vulnerabilities | group_by(.attack_type) | map({type: .[0].attack_type, count: length})'

# Extract just the endpoints
cat report.json | jq '.vulnerabilities[].endpoint'
```

### Automation Examples

```bash
# Run daily scan
echo "0 2 * * * /root/synexs/vuln_scanner.sh -u http://target.com -o /var/log/scans/\$(date +\%Y\%m\%d).json" | crontab -

# Alert on critical findings
./vuln_scanner.sh -u http://target.com -o scan.json
if grep -q '"severity": "Critical"' scan.json; then
    echo "CRITICAL vulnerabilities found!" | mail -s "Security Alert" admin@example.com
fi

# Compare two scans
./vuln_scanner.sh -u http://target.com -o before.json
# ... make fixes ...
./vuln_scanner.sh -u http://target.com -o after.json
diff <(jq -S '.vulnerabilities' before.json) <(jq -S '.vulnerabilities' after.json)
```

## Integration with CI/CD

### GitLab CI Example

```yaml
security_scan:
  stage: test
  script:
    - ./vuln_scanner.sh -u $STAGING_URL -o scan_results.json
    - |
      if grep -q '"severity": "Critical"' scan_results.json; then
        echo "Critical vulnerabilities found!"
        exit 1
      fi
  artifacts:
    paths:
      - scan_results.json
```

### GitHub Actions Example

```yaml
name: Security Scan
on: [push]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run vulnerability scan
        run: |
          chmod +x vuln_scanner.sh
          ./vuln_scanner.sh -u https://staging.example.com -o results.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: scan-results
          path: results.json
```

## Troubleshooting

### Script Won't Run

```bash
# Check bash version (need 4.0+)
bash --version

# Make sure it's executable
chmod +x vuln_scanner.sh

# Run with explicit bash
bash vuln_scanner.sh -u http://target.com
```

### curl Errors

```bash
# Check if curl is installed
which curl

# Install curl
apt install curl  # Debian/Ubuntu
yum install curl  # CentOS/RHEL
```

### No Color Output

```bash
# Force color output
export TERM=xterm-256color
./vuln_scanner.sh -u http://target.com
```

### JSON Report Issues

```bash
# Install jq for better reports
apt install jq

# Without jq, basic JSON is still created
./vuln_scanner.sh -u http://target.com -o report.json
```

## Customization

### Adding Custom Payloads

Edit the script and modify the payload arrays:

```bash
# Line ~200 - Add custom SQL payloads
local payloads=(
    "' OR '1'='1"
    "' OR '1'='1'--"
    "YOUR_CUSTOM_PAYLOAD_HERE"
)
```

### Adding New Endpoints

```bash
# Line ~210 - Add custom endpoints
local endpoints=(
    "/login/"
    "/search/"
    "/your-custom-endpoint/"
)
```

### Changing Timeout

```bash
# Command line
./vuln_scanner.sh -u http://target.com --timeout 30

# Or edit script default (line ~25)
TIMEOUT=30
```

## Performance Optimization

### Parallel Scanning

The bash version scans sequentially. For faster scanning, you can run multiple instances:

```bash
# Scan different types in parallel
./vuln_scanner.sh -u http://target.com -t sqli -o sqli.json &
./vuln_scanner.sh -u http://target.com -t xss -o xss.json &
./vuln_scanner.sh -u http://target.com -t rce -o rce.json &
wait

# Combine results
jq -s 'add' sqli.json xss.json rce.json > combined.json
```

### Reduce Timeout for Faster Scans

```bash
./vuln_scanner.sh -u http://target.com --timeout 5
```

## Security Considerations

1. **Authorization Required** - Always get written permission before scanning
2. **Rate Limiting** - The script doesn't rate limit, be careful with production systems
3. **Logging** - Scans may trigger IDS/IPS alerts
4. **Evidence Collection** - All findings include evidence snippets
5. **Responsible Disclosure** - Report vulnerabilities responsibly

## Best Practices

1. **Test in Staging First** - Don't scan production without approval
2. **Save All Reports** - Keep audit trail of scans
3. **Verify Manually** - Confirm automated findings
4. **Document Authorization** - Keep proof of permission
5. **Use VPN/Authorized IPs** - Scan from approved networks

## Comparison with Other Tools

### vs Nikto
- Bash scanner: Focused vulnerability testing
- Nikto: Comprehensive web server scanning

### vs OWASP ZAP
- Bash scanner: CLI, scriptable, lightweight
- ZAP: GUI, interactive, full-featured

### vs Burp Suite
- Bash scanner: Free, automated, simple
- Burp: Professional, manual testing, comprehensive

### vs SQLMap
- Bash scanner: Multi-vulnerability testing
- SQLMap: Deep SQL injection focus

## Legal Notice

**AUTHORIZED USE ONLY**

This tool is designed for:
- Authorized penetration testing
- Security assessments with permission
- Educational purposes
- CTF competitions
- Defensive security research

**Unauthorized use is illegal and unethical.**

## Support

- Python version: `/root/synexs/vuln_scanner.py`
- Bash version: `/root/synexs/vuln_scanner.sh`
- Documentation: `/root/synexs/SCANNER_README.md`
- Quick guide: `/root/synexs/VULNERABILITY_SCANNER_GUIDE.md`

## Contributing

To improve the bash scanner:

1. Add new payloads to existing scan functions
2. Create new scan functions following the pattern
3. Update the main() function to call new scans
4. Test thoroughly before deployment

## License

For authorized security testing and educational purposes only.

---

**Simple. Portable. Powerful. No dependencies required.**
