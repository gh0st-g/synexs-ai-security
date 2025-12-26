"""
LDAP Injection Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class LDAPScanner(BaseScanner):
    """Scanner for LDAP Injection vulnerabilities"""

    # LDAP injection payloads
    PAYLOADS = [
        "*",
        "*)(uid=*",
        "admin)(|(password=*))",
        "*)(uid=*))(|(uid=*",
        "*)(&(password=*)",
    ]

    def scan(self) -> List[ScanResult]:
        """Run LDAP injection scan"""
        results = []

        # Test LDAP endpoints
        endpoints = [
            '/ldap/',
            '/api/ldap/',
            '/auth/ldap/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for param in ['username', 'user', 'uid']:
                for payload in self.PAYLOADS:
                    result = self._test_ldap(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='ldap',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No LDAP injection vulnerabilities detected'
            ))

        return results

    def _test_ldap(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for LDAP injection"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if response and (
            'filter' in response.text.lower() or
            'uid=' in response.text or
            'ldap' in response.text.lower()
        ):
            return self.create_result(
                attack_type='ldap',
                endpoint=test_url,
                vulnerable=True,
                severity=Severity.MEDIUM,
                description=f'LDAP Injection vulnerability detected in {param} parameter',
                payload=payload,
                evidence=response.text[:500],
                recommendation='Escape LDAP special characters. Validate input against whitelist.',
                cwe='CWE-90',
                owasp='A03:2021 - Injection'
            )

        return None
