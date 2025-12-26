"""
Cross-Site Scripting (XSS) Scanner Module
"""

from typing import List
import html
from .scanner_base import BaseScanner, ScanResult, Severity


class XSSScanner(BaseScanner):
    """Scanner for XSS vulnerabilities"""

    # XSS payloads
    PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "javascript:alert('XSS')",
        "<iframe src='javascript:alert(1)'>",
        "'\"><script>alert(String.fromCharCode(88,83,83))</script>",
        "<body onload=alert('XSS')>",
        "<input autofocus onfocus=alert('XSS')>",
    ]

    def scan(self) -> List[ScanResult]:
        """Run XSS scan"""
        results = []

        # Test common endpoints
        endpoints = [
            '/comments/',
            '/search/',
            '/dashboard/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            # Test reflected XSS
            for param in ['q', 'search', 'highlight', 'message']:
                for payload in self.PAYLOADS:
                    result = self._test_reflected_xss(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

            # Test stored XSS (comments)
            if 'comment' in endpoint:
                for payload in self.PAYLOADS:
                    result = self._test_stored_xss(endpoint, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='xss',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No XSS vulnerabilities detected'
            ))

        return results

    def _test_reflected_xss(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for reflected XSS"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if response and payload in response.text:
            # Check if payload is unescaped
            if not html.escape(payload) in response.text:
                return self.create_result(
                    attack_type='xss',
                    endpoint=test_url,
                    vulnerable=True,
                    severity=Severity.HIGH,
                    description=f'Reflected XSS vulnerability detected in {param} parameter',
                    payload=payload,
                    evidence=response.text[:500] if response else None,
                    recommendation='Implement proper output encoding/escaping. Use Content-Security-Policy headers.',
                    cwe='CWE-79',
                    owasp='A03:2021 - Injection'
                )

        return None

    def _test_stored_xss(self, endpoint: str, payload: str) -> ScanResult:
        """Test for stored XSS in comments"""
        data = {
            'comment': payload
        }

        # Submit the payload
        post_response = self.make_request(endpoint, method='POST', data=data)

        if not post_response:
            return None

        # Retrieve the page to see if payload is stored
        get_response = self.make_request(endpoint)

        if get_response and payload in get_response.text:
            return self.create_result(
                attack_type='xss',
                endpoint=endpoint,
                vulnerable=True,
                severity=Severity.HIGH,
                description='Stored XSS vulnerability detected',
                payload=payload,
                evidence=get_response.text[:500],
                recommendation='Sanitize and escape user input before storing and displaying. Use CSP headers.',
                cwe='CWE-79',
                owasp='A03:2021 - Injection'
            )

        return None
