"""
Local File Inclusion (LFI) Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class LFIScanner(BaseScanner):
    """Scanner for Local File Inclusion vulnerabilities"""

    # LFI payloads
    PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\win.ini",
        "../../../../etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",
        "....//....//....//etc/passwd",
        "/etc/passwd",
        "file:///etc/passwd",
    ]

    # File indicators
    FILE_INDICATORS = [
        'root:x:0:0',
        '[extensions]',
        'for 16-bit app support',
        '# /etc/passwd',
    ]

    def scan(self) -> List[ScanResult]:
        """Run LFI scan"""
        results = []

        # Test file download endpoints
        endpoints = [
            '/download/',
            '/file/',
            '/api/file/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for param in ['file', 'filename', 'path', 'document']:
                for payload in self.PAYLOADS:
                    result = self._test_lfi(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='lfi',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No LFI vulnerabilities detected'
            ))

        return results

    def _test_lfi(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for LFI vulnerability"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if self.check_response(response, self.FILE_INDICATORS):
            return self.create_result(
                attack_type='lfi',
                endpoint=test_url,
                vulnerable=True,
                severity=Severity.HIGH,
                description=f'Local File Inclusion vulnerability detected in {param} parameter',
                payload=payload,
                evidence=response.text[:500] if response else None,
                recommendation='Validate and sanitize file paths. Use whitelist of allowed files. Never allow user input directly in file paths.',
                cwe='CWE-22',
                owasp='A01:2021 - Broken Access Control'
            )

        return None
