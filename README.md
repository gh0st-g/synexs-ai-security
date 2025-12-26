# 🛡️ Synexs AI Security Framework

A comprehensive web application security testing framework featuring honeypot deployment, attack detection, and professional vulnerability scanning tools.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bash](https://img.shields.io/badge/Bash-4.0+-green.svg)](https://www.gnu.org/software/bash/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## ⚠️ Legal Disclaimer

**FOR AUTHORIZED SECURITY TESTING ONLY**

This framework is designed for:
- ✅ Authorized penetration testing engagements
- ✅ Security assessments with written permission
- ✅ Educational purposes in controlled environments
- ✅ CTF (Capture The Flag) competitions
- ✅ Defensive security research

**Unauthorized access to computer systems is illegal.** Always obtain proper written authorization before scanning or testing any target.

## 🎯 Features

### 1. CryptoVault Honeypot
A realistic cryptocurrency wallet honeypot designed to attract and log attacks.

### 2. Dual Vulnerability Scanners

#### Interactive Bash Scanner ⭐ NEW!
- **Zero dependencies** - Only bash + curl required
- **Fully interactive** - No command-line arguments to remember
- **User-friendly** - Step-by-step guided prompts
- **Fast** - 0.1s startup time
- **Portable** - Works on any Linux/Unix system

#### Python Scanner
- Modular design with separate attack modules
- Native JSON support
- Robust error handling
- Easy to extend

### 3. Attack Detection Coverage

**11 vulnerability types detected:**
- SQL Injection (Critical)
- Cross-Site Scripting (High)
- Remote Code Execution (Critical)  
- Local File Inclusion (High)
- Server-Side Request Forgery (High)
- XML External Entity (High)
- Server-Side Template Injection (High)
- NoSQL Injection (High)
- LDAP Injection (Medium)
- GraphQL Introspection (Medium)
- HTTP Header Injection (Medium)

## 🚀 Quick Start

### Interactive Bash Scanner (Easiest!)

```bash
./vuln_scanner.sh
```

Then answer simple questions - that's it!

### Python Scanner

```bash
pip install -r scanner_requirements.txt
python3 vuln_scanner.py -u http://target.com
```

## 📁 Project Structure

```
synexs/
├── vuln_scanner.sh           # Interactive bash scanner ⭐
├── vuln_scanner.py           # Python scanner
├── scanner_modules/          # Python modules
├── cryptovault_honeypot/     # Django honeypot
└── docs/                     # Documentation
```

## 📖 Documentation

- **[Interactive Scanner Guide](INTERACTIVE_SCANNER_GUIDE.md)** - New interactive mode ⭐
- **[Quick Start Guide](VULNERABILITY_SCANNER_GUIDE.md)** - Getting started
- **[Scanner Comparison](SCANNER_COMPARISON.md)** - Python vs Bash
- **[Python Scanner Docs](SCANNER_README.md)** - Full documentation
- **[Bash Scanner Docs](BASH_SCANNER_README.md)** - Bash details

## 💡 Usage Examples

### Example 1: Interactive Scan

```bash
./vuln_scanner.sh

# Input examples:
example.com    # Target
1              # All scans
n              # No verbose  
n              # No file output
y              # Confirm
yes            # Authorization
```

### Example 2: Python Command-Line

```bash
python3 vuln_scanner.py -u http://target.com -v -o report.json
```

## 🔐 Security Best Practices

1. Always get written authorization
2. Test in staging environments first
3. Verify findings manually
4. Document everything
5. Use responsible disclosure

## 📝 Changelog

### v1.1 - Interactive Mode
- ✨ NEW: Fully interactive bash scanner
- ✨ NEW: No arguments needed
- 🐛 Fixed: IP sanitization
- 📚 Added: Interactive guide

### v1.0 - Initial Release
- ✅ Python & Bash scanners
- ✅ CryptoVault honeypot
- ✅ 11 vulnerability types

## 📄 License

MIT License - See LICENSE file

## 🌟 Contributing

Contributions welcome! Please read CONTRIBUTING.md

## 📧 Contact

- Issues: GitHub Issues
- Security: Report responsibly

---

**Use responsibly. Always get authorization. Help make the internet more secure.**
