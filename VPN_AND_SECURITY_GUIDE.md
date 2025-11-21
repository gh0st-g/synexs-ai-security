# SYNEXS VPN + Security Scanner Guide

## Problem Fixed ✅

**Original Issue:** VPN connection was routing ALL traffic (0.0.0.0/0) through VPN, causing:
- Loss of internet connectivity
- Unable to reach scan targets
- DNS resolution failures  
- Scan results not being saved

**Solution:** Created safe VPN connection scripts with:
- Connectivity checks before/after VPN connection
- Automatic DNS fallback
- Better error handling
- Results always saved even if scan fails

## Files Created

1. **`safe_vpn_connect.sh`** - Safe VPN connection with connectivity checks
2. **`safe_vpn_disconnect.sh`** - Safe VPN disconnection
3. **`synexs_security_scanner.sh`** - Updated scanner (already existed, now improved)

## Quick Start

### Test Your Setup

```bash
# 1. Test current internet connectivity
ping -c 3 8.8.8.8

# 2. Test VPN connection safely
/root/synexs/safe_vpn_connect.sh test

# 3. Connect to VPN
/root/synexs/safe_vpn_connect.sh connect

# 4. Disconnect from VPN
/root/synexs/safe_vpn_disconnect.sh
```

### Scan Your VPS (128.199.6.219 / synexs.net)

#### Option 1: WITHOUT VPN (Recommended for Your Own Servers)
```bash
./synexs_security_scanner.sh 128.199.6.219
# When asked about VPN, choose: n (no)
```

#### Option 2: WITH VPN (For Anonymous Scanning)
```bash
./synexs_security_scanner.sh synexs.net
# When asked about VPN, choose: y (yes)
```

## Usage Guide

### Safe VPN Scripts

**Connect to VPN:**
```bash
/root/synexs/safe_vpn_connect.sh
```

What it does:
- ✅ Tests internet before connecting
- ✅ Connects to VPN
- ✅ Tests internet after connecting
- ✅ Adds backup DNS if needed
- ✅ Shows your public IP

**Test Connectivity:**
```bash
/root/synexs/safe_vpn_connect.sh test
```

**Disconnect:**
```bash
/root/synexs/safe_vpn_disconnect.sh
```

### Security Scanner

**Run the scanner:**
```bash
cd /root/synexs
./synexs_security_scanner.sh
```

**Scan Types Available:**

1. **Fast Recon** (1-2 min) - Top 1000 ports
2. **Quick Web Test** - HTTP/HTTPS only
3. **Honeypot Test** - Test YOUR honeypot detection
4. **Version + OS Detection** - Identify services
5. **UDP Scan** - Top 200 UDP ports
6. **Vulnerability Scan** - Check for known vulns
7. **HTTP Deep Dive** - Comprehensive HTTP testing
8. **SSL/TLS Check** - Certificate and cipher analysis
9. **Full Aggressive** - All 65535 ports (slow!)
10. **Nuclear Option** - EVERYTHING (very slow!)

## Testing Your Honeypot

To test if your honeypot at 128.199.6.219 is detecting attacks:

```bash
# 1. Run scanner
./synexs_security_scanner.sh 128.199.6.219

# 2. Choose VPN option:
#    - 'n' for direct (faster)
#    - 'y' for anonymous

# 3. Select option 3: Honeypot Test

# 4. Check your honeypot logs:
ssh root@128.199.6.219
tail -f /root/synexs/datasets/honeypot/attacks.json
```

## Scan Results

All results saved to: `/root/synexs/scan_results/`

**View recent scans:**
```bash
ls -lht /root/synexs/scan_results/ | head -20
```

**View a specific scan:**
```bash
cat /root/synexs/scan_results/fast_recon_TIMESTAMP.txt
```

**Search for open ports:**
```bash
grep "open" /root/synexs/scan_results/*.txt
```

## Common Scenarios

### Scenario 1: Test Your Own VPS Security
```bash
# Direct connection (no VPN needed)
./synexs_security_scanner.sh 128.199.6.219
# Choose: n (no VPN)
# Select: 1 (Fast Recon)
```

### Scenario 2: Check if Ports are Open
```bash
# Quick web test
./synexs_security_scanner.sh synexs.net
# Choose: n (no VPN)
# Select: 2 (Quick Web Test)
```

### Scenario 3: Test Honeypot Detection
```bash
# Connect to VPN for realistic attack simulation
./synexs_security_scanner.sh 128.199.6.219
# Choose: y (use VPN)
# Select: 3 (Honeypot Test)

# Then check honeypot logs on the VPS
```

### Scenario 4: Full Security Audit
```bash
# For comprehensive testing
./synexs_security_scanner.sh 128.199.6.219
# Choose: n (no VPN for faster scans)
# Run scans: 1, 4, 6, 7, 8 in sequence
```

## Troubleshooting

### "Lost Internet After VPN Connection"

**Solution:** Use the new safe scripts:
```bash
/root/synexs/safe_vpn_disconnect.sh
/root/synexs/safe_vpn_connect.sh
```

The safe script tests connectivity and adds backup DNS automatically.

### "Can't Reach Target"

**Check:**
1. Is target reachable without VPN?
   ```bash
   ping 128.199.6.219
   ```

2. Is VPN blocking the target?
   ```bash
   # Disconnect VPN and try again
   /root/synexs/safe_vpn_disconnect.sh
   ping 128.199.6.219
   ```

3. Try scanning without VPN:
   ```bash
   # VPN not needed for your own servers
   ./synexs_security_scanner.sh 128.199.6.219
   # Choose: n
   ```

### "Scan Results Not Saved"

Results are ALWAYS saved to `/root/synexs/scan_results/` even if scan fails.

**Check results:**
```bash
ls -lht /root/synexs/scan_results/
```

**View partial results:**
```bash
cat /root/synexs/scan_results/[filename].txt
```

### "DNS Not Working"

The safe VPN script adds backup DNS automatically. Manual fix:
```bash
echo "nameserver 8.8.8.8" >> /etc/resolv.conf
echo "nameserver 1.1.1.1" >> /etc/resolv.conf
```

## Security Best Practices

✅ **DO:**
- Scan only YOUR own servers
- Use VPN for anonymous reconnaissance
- Scan without VPN for your own infrastructure (faster)
- Keep scan results for security documentation
- Test your honeypot regularly

❌ **DON'T:**
- Scan servers you don't own without permission
- Run nuclear scans during business hours (they're loud!)
- Scan production servers without alerting your team
- Forget to disconnect VPN when done

## Advanced Usage

### Automate Security Scans

Create a weekly scan:
```bash
# Add to crontab
crontab -e

# Run every Sunday at 2 AM
0 2 * * 0 /root/synexs/synexs_security_scanner.sh 128.199.6.219 <<< $'n\n1\n0\ny\n'
```

### Compare Scans Over Time

```bash
# Scan today
./synexs_security_scanner.sh 128.199.6.219
# Select: 1 (Fast Recon)

# Compare with previous scan
diff /root/synexs/scan_results/fast_recon_*.txt
```

### Export Results

```bash
# Copy all results
tar -czf security_scans_$(date +%Y%m%d).tar.gz /root/synexs/scan_results/

# Download to local machine
scp root@your-scanner:/root/security_scans_*.tar.gz .
```

## Quick Reference

```bash
# Safe VPN management
/root/synexs/safe_vpn_connect.sh        # Connect
/root/synexs/safe_vpn_connect.sh test   # Test
/root/synexs/safe_vpn_disconnect.sh     # Disconnect

# Scanner
cd /root/synexs
./synexs_security_scanner.sh <target>

# View results
ls -lht scan_results/
cat scan_results/[filename].txt
grep "open" scan_results/*.txt
```

## Support

If you encounter issues:

1. Check connectivity:
   ```bash
   /root/synexs/safe_vpn_connect.sh test
   ping 8.8.8.8
   ping 128.199.6.219
   ```

2. Check VPN status:
   ```bash
   ip addr show wg0
   ip route show
   ```

3. View scan logs:
   ```bash
   ls -lht /root/synexs/scan_results/
   ```

---

**Last Updated:** 2025-11-21
**Version:** 2.0 - Safe VPN Edition
