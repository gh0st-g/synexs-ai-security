"""
Jinja2 environment configuration
VULN: Intentionally insecure for SSTI testing
"""
from jinja2 import Environment


def environment(**options):
    """
    Create Jinja2 environment with intentionally unsafe settings
    """
    env = Environment(**options)
    # VULN: No autoescape - allows XSS and SSTI
    env.autoescape = False
    return env
