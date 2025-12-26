"""
Server-Side Request Forgery (SSRF) Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class SSRFScanner(BaseScanner):
    """Scanner for SSRF vulnerabilities"""

    # SSRF payloads
    PAYLOADS = [
        "http://localhost",
        "http://127.0.0.1",
        "http://0.0.0.0",
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        "file:///etc/passwd",
        "http://localhost:22",
        "http://[::1]",
    ]

    # SSRF indicators
    SSRF_INDICATORS = [
        'ssh-',
        'openssh',
        'root:x:0:0',
        'localhost',
        'loopback',
    ]

    def scan(self) -> List[ScanResult]:
        """Run SSRF scan"""
        results = []

        # Test SSRF-prone endpoints
        endpoints = [
            '/fetch-price/',
            '/api/fetch/',
            '/proxy/',
            '/webhook/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for param in ['url', 'uri', 'target', 'link']:
                for payload in self.PAYLOADS:
                    result = self._test_ssrf(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='ssrf',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No SSRF vulnerabilities detected'
            ))

        return results

    def _test_ssrf(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for SSRF vulnerability"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if response and (
            self.check_response(response, self.SSRF_INDICATORS) or
            'connection' in response.text.lower() or
            'localhost' in response.text.lower()
        ):
            return self.create_result(
                attack_type='ssrf',
                endpoint=test_url,
                vulnerable=True,
                severity=Severity.HIGH,
                description=f'SSRF vulnerability detected in {param} parameter',
                payload=payload,
                evidence=response.text[:500],
                recommendation='Validate and whitelist allowed URLs/domains. Disable redirects. Block private IP ranges.',
                cwe='CWE-918',
                owasp='A10:2021 - Server-Side Request Forgery'
            )

        return None
