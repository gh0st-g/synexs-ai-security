"""
SQL Injection Scanner Module
"""

from typing import List
from .scanner_base import BaseScanner, ScanResult, Severity


class SQLInjectionScanner(BaseScanner):
    """Scanner for SQL Injection vulnerabilities"""

    # SQL injection payloads
    PAYLOADS = [
        "' OR '1'='1",
        "' OR '1'='1'--",
        "' OR 1=1--",
        "admin'--",
        "' UNION SELECT NULL--",
        "' UNION SELECT NULL,NULL--",
        "' UNION SELECT NULL,NULL,NULL--",
        "1' AND '1'='2",
        "' AND SLEEP(5)--",
        "'; WAITFOR DELAY '00:00:05'--",
    ]

    # SQL error indicators
    SQL_ERRORS = [
        'sql syntax',
        'mysql',
        'sqlite',
        'postgresql',
        'ora-',
        'microsoft sql',
        'odbc',
        'jdbc',
        'warning: mysql',
        'syntax error',
        'unclosed quotation',
        'quoted string not properly terminated',
    ]

    def scan(self) -> List[ScanResult]:
        """Run SQL injection scan"""
        results = []

        # Test common endpoints
        endpoints = [
            '/login/',
            '/search/',
            '/api/user/',
            '/dashboard/',
        ]

        for endpoint in endpoints:
            self.log(f"Testing endpoint: {endpoint}")

            # Test GET parameters
            for param_name in ['q', 'id', 'user_id', 'username', 'search']:
                for payload in self.PAYLOADS:
                    result = self._test_sqli(endpoint, param_name, payload, method='GET')
                    if result:
                        results.append(result)
                        break  # Stop testing this param if vulnerable

            # Test POST parameters (login)
            if 'login' in endpoint:
                for payload in self.PAYLOADS:
                    result = self._test_sqli_post(endpoint, payload)
                    if result:
                        results.append(result)
                        break

        # If no vulnerabilities found, add negative result
        if not results:
            results.append(self.create_result(
                attack_type='sqli',
                endpoint=self.target_url,
                vulnerable=False,
                severity=Severity.INFO,
                description='No SQL injection vulnerabilities detected'
            ))

        return results

    def _test_sqli(self, endpoint: str, param: str, payload: str, method: str = 'GET') -> ScanResult:
        """Test for SQL injection via GET/POST parameter"""
        test_url = f"{endpoint}?{param}={payload}"

        response = self.make_request(test_url, method=method)

        if self.check_response(response, self.SQL_ERRORS):
            return self.create_result(
                attack_type='sqli',
                endpoint=test_url,
                vulnerable=True,
                severity=Severity.CRITICAL,
                description=f'SQL Injection vulnerability detected in {param} parameter',
                payload=payload,
                evidence=response.text[:500] if response else None,
                recommendation='Use parameterized queries/prepared statements. Never concatenate user input into SQL queries.',
                cwe='CWE-89',
                owasp='A03:2021 - Injection'
            )

        return None

    def _test_sqli_post(self, endpoint: str, payload: str) -> ScanResult:
        """Test for SQL injection in login form"""
        data = {
            'username': payload,
            'password': 'test'
        }

        response = self.make_request(endpoint, method='POST', data=data)

        if response and (
            self.check_response(response, self.SQL_ERRORS) or
            'dashboard' in response.url.lower() or
            response.status_code == 302
        ):
            return self.create_result(
                attack_type='sqli',
                endpoint=endpoint,
                vulnerable=True,
                severity=Severity.CRITICAL,
                description='SQL Injection vulnerability detected in login form',
                payload=payload,
                evidence=response.text[:500] if response else None,
                recommendation='Use parameterized queries/prepared statements for authentication.',
                cwe='CWE-89',
                owasp='A03:2021 - Injection'
            )

        return None
