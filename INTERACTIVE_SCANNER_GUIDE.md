# Interactive Bash Scanner - Quick Guide

## Super Simple Usage! 🚀

Just run the script - **no command-line arguments needed!**

```bash
./vuln_scanner.sh
```

That's it! The scanner will guide you through everything with interactive prompts.

## What Happens When You Run It

### Step 1: Enter Target URL
```
╔══════════════════════════════════════════════════════════════╗
║              Target Configuration                            ║
╚══════════════════════════════════════════════════════════════╝

Please enter the target URL or IP to scan:
Examples:
  - http://example.com
  - https://192.168.1.100
  - http://your-target.com

Target URL/IP:
```

**Just type the URL or IP:**
- `your-target.com` → Automatically becomes `http://your-target.com`
- `http://example.com` → Used as-is
- `https://secure.site` → Used as-is

### Step 2: Select Scan Types
```
╔══════════════════════════════════════════════════════════════╗
║              Select Scan Types                               ║
╚══════════════════════════════════════════════════════════════╝

Available scan types:

  [1] All scans (recommended)
  [2] SQL Injection only
  [3] XSS only
  [4] LFI only
  [5] RCE only
  [6] SSRF only
  [7] All injection attacks (SQLi, XSS, SSTI, NoSQLi)
  [8] Custom selection

Select option [1-8] (default: 1):
```

**Choose what to scan:**
- Press `1` or just `Enter` for all scans (recommended)
- Press `2` for quick SQL injection test
- Press `8` for custom selection

### Step 3: Configure Options
```
╔══════════════════════════════════════════════════════════════╗
║              Scan Options                                    ║
╚══════════════════════════════════════════════════════════════╝

Enable verbose mode? (y/n, default: n):
```

**Configure scan options:**
- Verbose mode: `y` for detailed debug output, `n` for clean output
- Save to file: `y` to save JSON report, `n` for screen only
- Timeout: Default is 10 seconds, increase for slow targets

### Step 4: Review and Confirm
```
╔══════════════════════════════════════════════════════════════╗
║              Scan Configuration Summary                      ║
╚══════════════════════════════════════════════════════════════╝

  Target:       http://your-target.com
  Scan Types:   all
  Verbose:      No
  Output File:  None
  Timeout:      10s

Continue with scan? (y/n):
```

**Last chance to review everything!**

### Step 5: Legal Authorization
```
╔══════════════════════════════════════════════════════════════╗
║                    LEGAL DISCLAIMER                          ║
╚══════════════════════════════════════════════════════════════╝

This tool is for AUTHORIZED security testing only.
Unauthorized access to computer systems is illegal.
By continuing, you confirm that you have proper authorization.

I have authorization to scan http://your-target.com (yes/no):
```

**Type `yes` to confirm authorization**

### Step 6: Scan Runs!
```
╔══════════════════════════════════════════════════════════════╗
║     Web Application Vulnerability Scanner v1.0              ║
║     For Authorized Security Testing Only                    ║
╚══════════════════════════════════════════════════════════════╝

Target: http://your-target.com
Scan Started: 2025-12-26 11:00:00

[1/11] Running SQL Injection scan...
  ✗ SQL Injection found: /login/?username=
...
```

## Example Session

Here's a complete example of scanning your honeypot:

```bash
$ ./vuln_scanner.sh

╔══════════════════════════════════════════════════════════════╗
║       Web Application Vulnerability Scanner (Bash)          ║
║       For Authorized Security Testing Only                  ║
╚══════════════════════════════════════════════════════════════╝

Please enter the target URL or IP to scan:
Examples:
  - http://example.com
  - https://192.168.1.100
  - http://your-target.com

Target URL/IP: your-target.com
Note: Added http:// prefix -> http://your-target.com
✓ Target set: http://your-target.com

╔══════════════════════════════════════════════════════════════╗
║              Select Scan Types                               ║
╚══════════════════════════════════════════════════════════════╝

Available scan types:
  [1] All scans (recommended)
  ...

Select option [1-8] (default: 1): 1
✓ Running all scans

╔══════════════════════════════════════════════════════════════╗
║              Scan Options                                    ║
╚══════════════════════════════════════════════════════════════╝

Enable verbose mode? (y/n, default: n): n
✓ Normal mode

Save results to JSON file? (y/n, default: n): y
Enter output filename (default: scan_results.json): honeypot_scan.json
✓ Results will be saved to: honeypot_scan.json

Request timeout in seconds (default: 10): 10
✓ Timeout set to: 10s

╔══════════════════════════════════════════════════════════════╗
║              Scan Configuration Summary                      ║
╚══════════════════════════════════════════════════════════════╝

  Target:       http://your-target.com
  Scan Types:   all
  Verbose:      No
  Output File:  honeypot_scan.json
  Timeout:      10s

Continue with scan? (y/n): y

╔══════════════════════════════════════════════════════════════╗
║                    LEGAL DISCLAIMER                          ║
╚══════════════════════════════════════════════════════════════╝

This tool is for AUTHORIZED security testing only.
Unauthorized access to computer systems is illegal.
By continuing, you confirm that you have proper authorization.

I have authorization to scan http://your-target.com (yes/no): yes
✓ Authorization confirmed

[Scan runs...]
```

## Quick Answers for Fast Scanning

Want to scan quickly? Here's what to type:

### Scan Your Honeypot (Fast)
1. Run: `./vuln_scanner.sh`
2. Type: `your-target.com` and press Enter
3. Press Enter (uses option 1 - all scans)
4. Type: `n` and press Enter (no verbose)
5. Type: `n` and press Enter (no file output)
6. Press Enter (default timeout)
7. Type: `y` and press Enter (confirm)
8. Type: `yes` and press Enter (authorization)

**Total: Just 7 key presses!**

### Quick SQL Injection Test
1. Run: `./vuln_scanner.sh`
2. Type target URL
3. Type: `2` (SQL injection only)
4. Type: `n`, `n`, Enter
5. Type: `y`, `yes`

## Advanced Usage

### Scanning Just Injection Attacks
- Choose option `7` when prompted for scan types
- Tests: SQLi, XSS, SSTI, NoSQLi

### Custom Scan Types
- Choose option `8`
- Enter comma-separated types: `sqli,xss,rce`

### Saving Results
- Answer `y` when asked about JSON output
- Enter filename or press Enter for default
- File is saved in current directory

## Comparison: Old vs New

### Old Way (Command Line Arguments)
```bash
./vuln_scanner.sh -u http://your-target.com -v -o report.json -t sqli,xss --timeout 15
```
**Problem:** Hard to remember all the flags!

### New Way (Interactive)
```bash
./vuln_scanner.sh
```
**Solution:** Just answer the questions!

## Benefits of Interactive Mode

✅ **No flags to remember** - The script asks you what it needs
✅ **Clear examples** - Shows you exactly what format to use
✅ **Validation** - Won't accept empty or invalid input
✅ **Auto-correction** - Adds `http://` if you forget
✅ **Summary screen** - Review everything before scanning
✅ **User-friendly** - Perfect for beginners
✅ **Still powerful** - All features available

## Tips

### Tip 1: Default Values
Most prompts have defaults. Just press Enter to use them:
- Scan types: `all` (option 1)
- Verbose: `no`
- Output file: `none`
- Timeout: `10s`

### Tip 2: Yes/No Questions
For y/n questions:
- `y`, `Y`, `yes`, `YES` all work for yes
- Anything else means no

### Tip 3: Quick Exit
Press `Ctrl+C` at any time to cancel

### Tip 4: URL Format
All these work:
- `example.com` → becomes `http://example.com`
- `192.168.1.1` → becomes `http://192.168.1.1`
- `http://site.com` → used as-is
- `https://secure.com` → used as-is

## Common Use Cases

### 1. Quick Test of Your Honeypot
```bash
./vuln_scanner.sh
# Type: your-target.com
# Press Enter 4 times (use all defaults)
# Type: y, yes
```

### 2. Full Scan with Report
```bash
./vuln_scanner.sh
# Enter target
# Choose option 1 (all scans)
# Type: n (no verbose)
# Type: y (save file)
# Enter filename
# Type: y, yes
```

### 3. Test Just SQL Injection
```bash
./vuln_scanner.sh
# Enter target
# Choose option 2 (SQL only)
# Type: n, n, Enter
# Type: y, yes
```

## Troubleshooting

### Script Exits Immediately
**Cause:** Missing dependencies (bash, curl)
**Solution:** Install curl: `apt install curl`

### No Color Output
**Cause:** Terminal doesn't support colors
**Solution:** Still works, just without colors

### Authorization Prompt Won't Accept "yes"
**Cause:** Need to type exactly "yes" (not "y")
**Solution:** Type the full word `yes`

## What Changed?

**Before:**
- Command-line arguments required
- Had to remember all flags
- Easy to make mistakes
- Less user-friendly

**After:**
- Fully interactive prompts
- Step-by-step guidance
- Clear examples
- Impossible to forget options

## Still Want Command-Line?

If you prefer the old command-line style, you can use the Python version:

```bash
python3 vuln_scanner.py -u http://target.com -v -o report.json
```

Both scanners have the same capabilities - choose what you prefer!

## Summary

**Old usage:**
```bash
./vuln_scanner.sh -u http://your-target.com -v -o report.json -t all --timeout 10
```

**New usage:**
```bash
./vuln_scanner.sh
```

Then just answer the questions! 🎉

---

**Easy. Simple. User-Friendly. No memorization required!**
