"""
HTTP Header Injection Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class HeaderInjectionScanner(BaseScanner):
    """Scanner for HTTP Header Injection and related vulnerabilities"""

    # Header injection payloads
    PAYLOADS = [
        "%0d%0aSet-Cookie: injected=true",
        "\\r\\nX-Injected: true",
        "\r\nX-Injected: true",
    ]

    def scan(self) -> List[ScanResult]:
        """Run header injection scan"""
        results = []

        # Test endpoints that set headers
        endpoints = [
            '/reset-password/',
            '/api/redirect/',
            '/logout/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for param in ['email', 'redirect', 'next']:
                for payload in self.PAYLOADS:
                    result = self._test_header_injection(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='header_injection',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No header injection vulnerabilities detected'
            ))

        return results

    def _test_header_injection(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for header injection"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if response:
            # Check for injected header
            if 'X-Injected' in response.headers or 'injected' in str(response.headers).lower():
                return self.create_result(
                    attack_type='header_injection',
                    endpoint=test_url,
                    vulnerable=True,
                    severity=Severity.MEDIUM,
                    description=f'Header Injection vulnerability detected in {param} parameter',
                    payload=payload,
                    evidence=str(response.headers)[:500],
                    recommendation='Sanitize user input before using in HTTP headers. Validate and encode CRLF characters.',
                    cwe='CWE-113',
                    owasp='A03:2021 - Injection'
                )

        return None
