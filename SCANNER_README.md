# Web Application Vulnerability Scanner

A professional-grade vulnerability scanner for defensive security testing and penetration testing engagements.

## ⚠️ LEGAL DISCLAIMER

**AUTHORIZED USE ONLY**

This tool is designed for:
- Authorized penetration testing engagements
- Security assessments with written authorization
- Educational purposes in controlled environments
- CTF competitions
- Defensive security research

**Unauthorized access to computer systems is illegal.** Always obtain proper authorization before scanning any target.

## Features

### Supported Vulnerability Types

The scanner detects the following web application vulnerabilities:

1. **SQL Injection (SQLi)**
   - Classic SQL injection
   - Blind SQL injection
   - Union-based SQLi
   - Error-based SQLi

2. **Cross-Site Scripting (XSS)**
   - Reflected XSS
   - Stored XSS
   - DOM-based XSS
   - Multiple payload vectors

3. **Local File Inclusion (LFI)**
   - Path traversal
   - Directory traversal
   - File disclosure

4. **Remote Code Execution (RCE)**
   - Command injection
   - OS command execution
   - Shell injection

5. **Server-Side Request Forgery (SSRF)**
   - Internal network access
   - Cloud metadata access
   - Port scanning

6. **XML External Entity (XXE)**
   - File disclosure via XXE
   - SSRF via XXE

7. **Server-Side Template Injection (SSTI)**
   - Jinja2 injection
   - Template engine exploitation

8. **NoSQL Injection**
   - MongoDB injection
   - Operator injection

9. **LDAP Injection**
   - Filter injection
   - Authentication bypass

10. **GraphQL Vulnerabilities**
    - Introspection enabled
    - Schema disclosure

11. **HTTP Header Injection**
    - Response splitting
    - CRLF injection

## Installation

1. Install Python dependencies:

```bash
pip install -r scanner_requirements.txt
```

2. Make the scanner executable:

```bash
chmod +x vuln_scanner.py
```

## Usage

### Basic Scan (All Vulnerability Types)

```bash
python3 vuln_scanner.py -u http://target.com
```

### Scan Specific Vulnerability Types

```bash
python3 vuln_scanner.py -u http://target.com -t sqli xss
```

### Verbose Mode

```bash
python3 vuln_scanner.py -u http://target.com -v
```

### Export Results to JSON

```bash
python3 vuln_scanner.py -u http://target.com -o report.json
```

### Custom Timeout

```bash
python3 vuln_scanner.py -u http://target.com --timeout 15
```

## Scan Types

Use the `-t` flag to specify which scans to run:

| Scan Type | Description |
|-----------|-------------|
| `sqli` | SQL Injection |
| `xss` | Cross-Site Scripting |
| `lfi` | Local File Inclusion |
| `rce` | Remote Code Execution |
| `ssrf` | Server-Side Request Forgery |
| `xxe` | XML External Entity |
| `ssti` | Server-Side Template Injection |
| `nosqli` | NoSQL Injection |
| `ldap` | LDAP Injection |
| `graphql` | GraphQL Vulnerabilities |
| `headers` | HTTP Header Injection |
| `all` | Run all scans (default) |

## Output Format

### Console Output

The scanner provides:
- Real-time scan progress
- Color-coded severity levels
- Detailed vulnerability findings
- Remediation recommendations

### JSON Report

Export findings to JSON for integration with other tools:

```json
{
  "scan_info": {
    "target": "http://target.com",
    "scan_start": "2025-01-15T10:30:00",
    "scan_end": "2025-01-15T10:35:00",
    "duration_seconds": 300
  },
  "summary": {
    "total_vulnerabilities": 5,
    "critical": 2,
    "high": 2,
    "medium": 1,
    "low": 0
  },
  "vulnerabilities": [
    {
      "attack_type": "sqli",
      "severity": "Critical",
      "endpoint": "/login/?username=' OR '1'='1",
      "description": "SQL Injection vulnerability detected",
      "payload": "' OR '1'='1",
      "evidence": "Database error: ...",
      "recommendation": "Use parameterized queries",
      "cwe": "CWE-89",
      "owasp": "A03:2021 - Injection"
    }
  ]
}
```

## Severity Levels

- **Critical**: Immediate exploitation possible (RCE, SQLi)
- **High**: Serious security impact (SSRF, XXE, SSTI)
- **Medium**: Moderate risk (XSS, LFI, LDAP, GraphQL)
- **Low**: Information disclosure or minor issues

## Example Workflows

### Security Assessment

```bash
# Full scan of target application
python3 vuln_scanner.py -u https://app.example.com -o assessment_report.json

# Review findings
cat assessment_report.json | jq '.vulnerabilities[] | select(.severity=="Critical")'
```

### Testing Honeypot

```bash
# Scan the CryptoVault honeypot
python3 vuln_scanner.py -u http://your-target.com -v -o honeypot_scan.json
```

### Focused Testing

```bash
# Test only injection vulnerabilities
python3 vuln_scanner.py -u http://target.com -t sqli xss ssti nosqli

# Test only file-related vulnerabilities
python3 vuln_scanner.py -u http://target.com -t lfi xxe
```

## Architecture

```
vuln_scanner.py                 # Main scanner orchestrator
├── scanner_modules/
│   ├── scanner_base.py         # Base classes and utilities
│   ├── sqli_scanner.py         # SQL Injection detection
│   ├── xss_scanner.py          # XSS detection
│   ├── lfi_scanner.py          # LFI detection
│   ├── rce_scanner.py          # RCE detection
│   ├── ssrf_scanner.py         # SSRF detection
│   ├── xxe_scanner.py          # XXE detection
│   ├── ssti_scanner.py         # SSTI detection
│   ├── nosqli_scanner.py       # NoSQL Injection detection
│   ├── ldap_scanner.py         # LDAP Injection detection
│   ├── graphql_scanner.py      # GraphQL vulnerability detection
│   └── header_scanner.py       # Header Injection detection
```

## Extending the Scanner

### Adding a New Scanner Module

1. Create a new scanner file in `scanner_modules/`:

```python
from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity

class MyScanner(BaseScanner):
    def scan(self) -> List[ScanResult]:
        results = []
        # Implement scan logic
        return results
```

2. Import in `vuln_scanner.py`:

```python
from scanner_modules.my_scanner import MyScanner
```

3. Add to scanners dict:

```python
self.scanners = {
    # ...
    'myscan': MyScanner(target_url, verbose),
}
```

## Best Practices

1. **Always obtain authorization** before scanning
2. **Use rate limiting** to avoid overwhelming targets
3. **Review findings carefully** - verify vulnerabilities manually
4. **Export reports** for documentation and tracking
5. **Run in isolated environments** for testing
6. **Update payloads** regularly based on new attack vectors

## Integration with Honeypot

This scanner is designed to work with the CryptoVault honeypot:

```bash
# Scan the honeypot to verify vulnerabilities are logged
python3 vuln_scanner.py -u http://your-target.com -v

# Check honeypot dashboard for attack logs
# Visit: http://your-target.com/admin/?role=admin
```

## Additional Attack Types to Consider

Based on your requirements, here are additional vulnerability types you can implement:

- **Cache Poisoning**: Test X-Forwarded-Host and cache control headers
- **HTTP Parameter Pollution (HPP)**: Duplicate parameter injection
- **Polyglot Payloads**: Multi-context injection vectors
- **IDOR**: Insecure Direct Object References
- **CSRF**: Cross-Site Request Forgery
- **Deserialization**: Unsafe object deserialization
- **Authentication Bypass**: Broken authentication
- **Session Management**: Session fixation, hijacking

## Troubleshooting

### SSL Certificate Errors

The scanner disables SSL verification by default for testing. To enable:

```python
# In scanner_base.py
self.session.verify = True
```

### Timeout Issues

Increase timeout for slow targets:

```bash
python3 vuln_scanner.py -u http://target.com --timeout 30
```

### False Positives

Always verify findings manually:
1. Review the evidence field
2. Test payloads manually in a browser
3. Check for actual exploitability

## License

For authorized security testing and educational purposes only.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Scanner modules inherit from BaseScanner
- Payloads are well-tested
- Documentation is updated

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review scanner logs with `-v` flag
3. Verify target is accessible
4. Ensure proper authorization is obtained

---

**Remember: With great power comes great responsibility. Use ethically.**
