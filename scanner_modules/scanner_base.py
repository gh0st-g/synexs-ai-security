"""
Base scanner class and data structures
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import requests
import urllib3
from urllib.parse import urljoin, urlparse

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Severity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"


@dataclass
class ScanResult:
    """Represents a scan result"""
    attack_type: str
    endpoint: str
    vulnerable: bool
    severity: Severity
    description: str
    payload: Optional[str] = None
    evidence: Optional[str] = None
    recommendation: str = ""
    cwe: Optional[str] = None
    owasp: Optional[str] = None


class BaseScanner:
    """
    Base class for all vulnerability scanners
    """

    def __init__(self, target_url: str, verbose: bool = False, timeout: int = 10):
        self.target_url = target_url.rstrip('/')
        self.verbose = verbose
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(f"    [DEBUG] {message}")

    def make_request(self, endpoint: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """
        Make HTTP request with error handling

        Args:
            endpoint: Target endpoint
            method: HTTP method
            **kwargs: Additional arguments for requests

        Returns:
            Response object or None if error
        """
        url = urljoin(self.target_url, endpoint)

        try:
            if 'timeout' not in kwargs:
                kwargs['timeout'] = self.timeout

            response = self.session.request(method, url, **kwargs)
            return response

        except requests.exceptions.Timeout:
            self.log(f"Timeout accessing {url}")
            return None
        except requests.exceptions.ConnectionError:
            self.log(f"Connection error accessing {url}")
            return None
        except Exception as e:
            self.log(f"Error accessing {url}: {e}")
            return None

    def check_response(self, response: Optional[requests.Response], indicators: List[str]) -> bool:
        """
        Check if response contains vulnerability indicators

        Args:
            response: HTTP response
            indicators: List of strings to search for in response

        Returns:
            True if any indicator found
        """
        if not response:
            return False

        response_text = response.text.lower()

        for indicator in indicators:
            if indicator.lower() in response_text:
                return True

        return False

    def scan(self) -> List[ScanResult]:
        """
        Run the scan - to be implemented by subclasses

        Returns:
            List of ScanResult objects
        """
        raise NotImplementedError("Subclasses must implement scan()")

    def create_result(
        self,
        attack_type: str,
        endpoint: str,
        vulnerable: bool,
        severity: Severity,
        description: str,
        payload: Optional[str] = None,
        evidence: Optional[str] = None,
        recommendation: str = "",
        cwe: Optional[str] = None,
        owasp: Optional[str] = None
    ) -> ScanResult:
        """Helper method to create scan results"""
        return ScanResult(
            attack_type=attack_type,
            endpoint=endpoint,
            vulnerable=vulnerable,
            severity=severity,
            description=description,
            payload=payload,
            evidence=evidence,
            recommendation=recommendation,
            cwe=cwe,
            owasp=owasp
        )
