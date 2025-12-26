"""
Remote Code Execution (RCE) Scanner Module
"""

from typing import List
import time
from .scanner_base import BaseScanner, ScanResult, Severity


class RCEScanner(BaseScanner):
    """Scanner for Remote Code Execution vulnerabilities"""

    # RCE payloads
    PAYLOADS = [
        "whoami",
        "id",
        "cat /etc/passwd",
        "; ls",
        "| whoami",
        "&& whoami",
        "`whoami`",
        "$(whoami)",
    ]

    # Command execution indicators
    RCE_INDICATORS = [
        'uid=',
        'gid=',
        'groups=',
        'root:x:0:0',
        'www-data',
        'nginx',
        'apache',
    ]

    def scan(self) -> List[ScanResult]:
        """Run RCE scan"""
        results = []

        # Test command execution endpoints
        endpoints = [
            '/system-check/',
            '/api/exec/',
            '/debug/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for param in ['cmd', 'command', 'exec', 'run']:
                for payload in self.PAYLOADS:
                    result = self._test_rce(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='rce',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No RCE vulnerabilities detected'
            ))

        return results

    def _test_rce(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for RCE vulnerability"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if self.check_response(response, self.RCE_INDICATORS):
            return self.create_result(
                attack_type='rce',
                endpoint=test_url,
                vulnerable=True,
                severity=Severity.CRITICAL,
                description=f'Remote Code Execution vulnerability detected in {param} parameter',
                payload=payload,
                evidence=response.text[:500] if response else None,
                recommendation='Never execute user input as system commands. Use whitelisting and input validation. Avoid shell=True in subprocess calls.',
                cwe='CWE-78',
                owasp='A03:2021 - Injection'
            )

        return None
