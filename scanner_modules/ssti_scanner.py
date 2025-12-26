"""
Server-Side Template Injection (SSTI) Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class SSTIScanner(BaseScanner):
    """Scanner for SSTI vulnerabilities"""

    # SSTI payloads (Jinja2, Twig, etc.)
    PAYLOADS = [
        "{{7*7}}",
        "{{7*'7'}}",
        "${7*7}",
        "{{config}}",
        "{{self}}",
        "{{''.__class__.__mro__[1].__subclasses__()}}",
        "{%print('test')%}",
    ]

    def scan(self) -> List[ScanResult]:
        """Run SSTI scan"""
        results = []

        # Test template rendering endpoints
        endpoints = [
            '/render/',
            '/template/',
            '/preview/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for param in ['template', 'name', 'content']:
                for payload in self.PAYLOADS:
                    result = self._test_ssti(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='ssti',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No SSTI vulnerabilities detected'
            ))

        return results

    def _test_ssti(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for SSTI vulnerability"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if response:
            # Check for successful evaluation
            if payload == "{{7*7}}" and "49" in response.text:
                return self.create_result(
                    attack_type='ssti',
                    endpoint=test_url,
                    vulnerable=True,
                    severity=Severity.HIGH,
                    description=f'SSTI vulnerability detected in {param} parameter',
                    payload=payload,
                    evidence=response.text[:500],
                    recommendation='Never render user-controlled template strings. Use sandboxed template engines.',
                    cwe='CWE-94',
                    owasp='A03:2021 - Injection'
                )
            elif payload == "{{7*'7'}}" and "7777777" in response.text:
                return self.create_result(
                    attack_type='ssti',
                    endpoint=test_url,
                    vulnerable=True,
                    severity=Severity.HIGH,
                    description=f'SSTI vulnerability detected in {param} parameter',
                    payload=payload,
                    evidence=response.text[:500],
                    recommendation='Never render user-controlled template strings. Use sandboxed template engines.',
                    cwe='CWE-94',
                    owasp='A03:2021 - Injection'
                )

        return None
