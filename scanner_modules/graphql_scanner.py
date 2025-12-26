"""
GraphQL Injection Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class GraphQLScanner(BaseScanner):
    """Scanner for GraphQL vulnerabilities"""

    # GraphQL introspection query
    INTROSPECTION_QUERY = '''
    {
        __schema {
            types {
                name
            }
        }
    }
    '''

    def scan(self) -> List[ScanResult]:
        """Run GraphQL scan"""
        results = []

        # Test GraphQL endpoints
        endpoints = [
            '/graphql/',
            '/api/graphql/',
            '/v1/graphql/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            result = self._test_introspection(endpoint)
            if result:
                results.append(result)

        if not results:
            results.append(self.create_result(
                attack_type='graphql',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No GraphQL vulnerabilities detected'
            ))

        return results

    def _test_introspection(self, endpoint: str) -> ScanResult:
        """Test for GraphQL introspection enabled"""
        data = {'query': self.INTROSPECTION_QUERY}

        response = self.make_request(endpoint, method='POST', json=data)

        if response and '__schema' in response.text:
            return self.create_result(
                attack_type='graphql',
                endpoint=endpoint,
                vulnerable=True,
                severity=Severity.MEDIUM,
                description='GraphQL introspection is enabled',
                payload=self.INTROSPECTION_QUERY[:200],
                evidence=response.text[:500],
                recommendation='Disable introspection in production. Implement query depth limiting and rate limiting.',
                cwe='CWE-200',
                owasp='A01:2021 - Broken Access Control'
            )

        return None
