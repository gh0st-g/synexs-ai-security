#!/usr/bin/env python3
"""
Professional Web Application Vulnerability Scanner
For Defensive Security Testing and Penetration Testing

AUTHORIZED USE ONLY:
- Penetration testing engagements
- Security assessments with written authorization
- Educational purposes
- CTF competitions
- Defensive security research

Author: Purple Team Security Framework
"""

import argparse
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from colorama import Fore, Style, init

from scanner_modules.sqli_scanner import SQLInjectionScanner
from scanner_modules.xss_scanner import XSSScanner
from scanner_modules.lfi_scanner import LFIScanner
from scanner_modules.rce_scanner import RCEScanner
from scanner_modules.ssrf_scanner import SSRFScanner
from scanner_modules.xxe_scanner import XXEScanner
from scanner_modules.ssti_scanner import SSTIScanner
from scanner_modules.nosqli_scanner import NoSQLInjectionScanner
from scanner_modules.ldap_scanner import LDAPScanner
from scanner_modules.graphql_scanner import GraphQLScanner
from scanner_modules.header_scanner import HeaderInjectionScanner
from scanner_modules.scanner_base import ScanResult, Severity

# Initialize colorama
init(autoreset=True)


class VulnerabilityScanner:
    """
    Main vulnerability scanner class that orchestrates all scan modules
    """

    def __init__(self, target_url: str, verbose: bool = False):
        self.target_url = target_url.rstrip('/')
        self.verbose = verbose
        self.results: List[ScanResult] = []
        self.start_time = None
        self.end_time = None

        # Initialize all scanner modules
        self.scanners = {
            'sqli': SQLInjectionScanner(target_url, verbose),
            'xss': XSSScanner(target_url, verbose),
            'lfi': LFIScanner(target_url, verbose),
            'rce': RCEScanner(target_url, verbose),
            'ssrf': SSRFScanner(target_url, verbose),
            'xxe': XXEScanner(target_url, verbose),
            'ssti': SSTIScanner(target_url, verbose),
            'nosqli': NoSQLInjectionScanner(target_url, verbose),
            'ldap': LDAPScanner(target_url, verbose),
            'graphql': GraphQLScanner(target_url, verbose),
            'headers': HeaderInjectionScanner(target_url, verbose),
        }

    def print_banner(self):
        """Print scanner banner"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║     Web Application Vulnerability Scanner v1.0              ║
║     For Authorized Security Testing Only                    ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}Target: {self.target_url}
Scan Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}
"""
        print(banner)

    def run_scan(self, scan_types: Optional[List[str]] = None) -> List[ScanResult]:
        """
        Run vulnerability scans

        Args:
            scan_types: List of scan types to run. If None, run all scans.

        Returns:
            List of ScanResult objects
        """
        self.start_time = time.time()
        self.print_banner()

        # Determine which scans to run
        if scan_types is None or 'all' in scan_types:
            scanners_to_run = self.scanners.items()
        else:
            scanners_to_run = [(k, v) for k, v in self.scanners.items() if k in scan_types]

        total_scans = len(scanners_to_run)
        current_scan = 0

        # Run each scanner
        for scan_name, scanner in scanners_to_run:
            current_scan += 1
            print(f"\n{Fore.CYAN}[{current_scan}/{total_scans}] Running {scan_name.upper()} scan...{Style.RESET_ALL}")

            try:
                scan_results = scanner.scan()
                self.results.extend(scan_results)

                # Print results summary for this scan
                vulnerabilities = [r for r in scan_results if r.vulnerable]
                if vulnerabilities:
                    print(f"{Fore.RED}  ✗ Found {len(vulnerabilities)} vulnerability(ies){Style.RESET_ALL}")
                else:
                    print(f"{Fore.GREEN}  ✓ No vulnerabilities found{Style.RESET_ALL}")

            except Exception as e:
                print(f"{Fore.RED}  ✗ Error running {scan_name} scan: {e}{Style.RESET_ALL}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()

        self.end_time = time.time()
        return self.results

    def print_summary(self):
        """Print scan summary"""
        vulnerabilities = [r for r in self.results if r.vulnerable]

        # Count by severity
        critical = sum(1 for v in vulnerabilities if v.severity == Severity.CRITICAL)
        high = sum(1 for v in vulnerabilities if v.severity == Severity.HIGH)
        medium = sum(1 for v in vulnerabilities if v.severity == Severity.MEDIUM)
        low = sum(1 for v in vulnerabilities if v.severity == Severity.LOW)

        scan_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0

        summary = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                     SCAN SUMMARY                             ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}Total Vulnerabilities Found: {len(vulnerabilities)}{Style.RESET_ALL}

{Fore.RED}  Critical: {critical}{Style.RESET_ALL}
{Fore.MAGENTA}  High:     {high}{Style.RESET_ALL}
{Fore.YELLOW}  Medium:   {medium}{Style.RESET_ALL}
{Fore.CYAN}  Low:      {low}{Style.RESET_ALL}

{Fore.CYAN}Scan Duration: {scan_duration:.2f} seconds{Style.RESET_ALL}
"""
        print(summary)

        # Print detailed findings
        if vulnerabilities:
            print(f"\n{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗")
            print(f"║                  DETAILED FINDINGS                           ║")
            print(f"╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

            for idx, vuln in enumerate(vulnerabilities, 1):
                severity_color = {
                    Severity.CRITICAL: Fore.RED,
                    Severity.HIGH: Fore.MAGENTA,
                    Severity.MEDIUM: Fore.YELLOW,
                    Severity.LOW: Fore.CYAN,
                }.get(vuln.severity, Fore.WHITE)

                print(f"{Fore.WHITE}[{idx}] {vuln.attack_type.upper()}{Style.RESET_ALL}")
                print(f"    {severity_color}Severity: {vuln.severity.value}{Style.RESET_ALL}")
                print(f"    Endpoint: {vuln.endpoint}")
                print(f"    Description: {vuln.description}")
                if vuln.payload:
                    print(f"    Payload: {vuln.payload[:100]}...")
                if vuln.evidence:
                    print(f"    Evidence: {vuln.evidence[:200]}...")
                print(f"    Recommendation: {vuln.recommendation}")
                print()

    def export_json(self, output_file: str):
        """Export results to JSON"""
        vulnerabilities = [r for r in self.results if r.vulnerable]

        report = {
            'scan_info': {
                'target': self.target_url,
                'scan_start': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'scan_end': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'duration_seconds': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            },
            'summary': {
                'total_vulnerabilities': len(vulnerabilities),
                'critical': sum(1 for v in vulnerabilities if v.severity == Severity.CRITICAL),
                'high': sum(1 for v in vulnerabilities if v.severity == Severity.HIGH),
                'medium': sum(1 for v in vulnerabilities if v.severity == Severity.MEDIUM),
                'low': sum(1 for v in vulnerabilities if v.severity == Severity.LOW),
            },
            'vulnerabilities': [
                {
                    'attack_type': v.attack_type,
                    'severity': v.severity.value,
                    'endpoint': v.endpoint,
                    'description': v.description,
                    'payload': v.payload,
                    'evidence': v.evidence,
                    'recommendation': v.recommendation,
                    'cwe': v.cwe,
                    'owasp': v.owasp,
                }
                for v in vulnerabilities
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"{Fore.GREEN}Report exported to: {output_file}{Style.RESET_ALL}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Web Application Vulnerability Scanner - For Authorized Security Testing Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -u http://target.com                    # Scan all vulnerabilities
  %(prog)s -u http://target.com -t sqli xss        # Scan specific types
  %(prog)s -u http://target.com -o report.json     # Export to JSON
  %(prog)s -u http://target.com -v                 # Verbose mode

Scan Types:
  sqli      - SQL Injection
  xss       - Cross-Site Scripting
  lfi       - Local File Inclusion
  rce       - Remote Code Execution
  ssrf      - Server-Side Request Forgery
  xxe       - XML External Entity
  ssti      - Server-Side Template Injection
  nosqli    - NoSQL Injection
  ldap      - LDAP Injection
  graphql   - GraphQL Injection
  headers   - HTTP Header Injection

LEGAL NOTICE:
This tool is for authorized security testing only. Unauthorized access to
computer systems is illegal. Ensure you have proper authorization before
scanning any target.
        """
    )

    parser.add_argument('-u', '--url', required=True, help='Target URL')
    parser.add_argument('-t', '--types', nargs='+', help='Scan types (default: all)')
    parser.add_argument('-o', '--output', help='Output JSON report file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout (default: 10)')

    args = parser.parse_args()

    # Legal disclaimer
    print(f"{Fore.RED}{'='*70}")
    print("LEGAL DISCLAIMER: This tool is for AUTHORIZED security testing only.")
    print("Unauthorized access to computer systems is illegal.")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    response = input(f"{Fore.YELLOW}Do you have authorization to scan {args.url}? (yes/no): {Style.RESET_ALL}")
    if response.lower() not in ['yes', 'y']:
        print(f"{Fore.RED}Scan cancelled. Authorization required.{Style.RESET_ALL}")
        sys.exit(1)

    # Initialize and run scanner
    scanner = VulnerabilityScanner(args.url, verbose=args.verbose)
    scanner.run_scan(scan_types=args.types)
    scanner.print_summary()

    # Export if requested
    if args.output:
        scanner.export_json(args.output)


if __name__ == '__main__':
    main()
