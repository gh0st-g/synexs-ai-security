"""
Attack Logger Middleware - Logs all requests
"""
import logging
import re
from django.utils.deprecation import MiddlewareMixin
from .models import AttackLog

logger = logging.getLogger('vault.attacks')


class AttackLoggerMiddleware(MiddlewareMixin):
    """
    Log every HTTP request and detect potential attacks
    """

    # Attack pattern detection
    ATTACK_PATTERNS = {
        'sqli': [
            r"(\bor\b|\band\b).*?=.*?",
            r"union.*?select",
            r"drop\s+table",
            r"insert\s+into",
            r"--",
            r"';",
            r"1=1",
            r"1' or '1'='1",
        ],
        'xss': [
            r"<script",
            r"javascript:",
            r"onerror=",
            r"onload=",
            r"<img.*?src",
            r"alert\(",
            r"document\.cookie",
        ],
        'lfi': [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e",
            r"/etc/passwd",
            r"/etc/shadow",
            r"c:\\windows",
        ],
        'rce': [
            r"\|\s*\w+",
            r";\s*\w+",
            r"`.*?`",
            r"\$\(",
            r"&amp;&amp;",
            r"wget\s",
            r"curl\s",
        ],
        'xxe': [
            r"<!ENTITY",
            r"SYSTEM",
            r"<!DOCTYPE",
            r"file://",
        ],
        'nosqli': [
            r"\$ne",
            r"\$gt",
            r"\$where",
            r"\$regex",
        ],
        'ssti': [
            r"{{.*?}}",
            r"{%.*?%}",
            r"${.*?}",
            r"__class__",
            r"__mro__",
        ],
        'ssrf': [
            r"localhost",
            r"127\.0\.0\.1",
            r"0\.0\.0\.0",
            r"169\.254",
            r"::1",
            r"@",
            r"file://",
            r"dict://",
            r"gopher://",
        ],
        'graphql': [
            r"__schema",
            r"__type",
            r"introspectionquery",
            r"query.*{.*}",
            r"mutation.*{.*}",
        ],
        'ldap': [
            r"\*\)\(",
            r"\)\(",
            r"\|\(",
            r"&\(",
            r"cn=",
            r"uid=",
        ],
        'hpp': [
            r"[\?&]\w+=.*&\1=",  # Duplicate parameter names
        ],
        'header_injection': [
            r"%0d%0a",
            r"\\r\\n",
            r"\r\n",
            r"\\n",
            r"set-cookie:",
            r"location:",
        ],
        'cache_poisoning': [
            r"x-forwarded-host:",
            r"x-forwarded-scheme:",
            r"x-original-url:",
            r"x-rewrite-url:",
        ],
        'polyglot': [
            r"javas\w*:",
            r"'\"<>",
            r"javascript:.*<.*>",
            r"<.*on\w+=.*>",
        ],
    }

    def get_client_ip(self, request):
        """Extract real client IP"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    def detect_attack(self, text):
        """
        Detect attack patterns in text
        Returns: (attack_type, severity, payload)
        """
        if not text:
            return None, 'low', None

        text_lower = text.lower()
        detected = []

        for attack_type, patterns in self.ATTACK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected.append(attack_type)
                    break

        if not detected:
            return None, 'low', None

        # Determine severity
        severity = 'low'
        if 'rce' in detected or 'sqli' in detected:
            severity = 'critical'
        elif 'xxe' in detected or 'ssti' in detected or 'nosqli' in detected or 'ssrf' in detected:
            severity = 'high'
        elif 'xss' in detected or 'lfi' in detected or 'ldap' in detected or 'graphql' in detected:
            severity = 'medium'
        elif 'hpp' in detected or 'header_injection' in detected or 'cache_poisoning' in detected:
            severity = 'medium'

        return ', '.join(detected), severity, text[:1000]

    def process_request(self, request):
        """Log request before processing"""
        try:
            ip = self.get_client_ip(request)
            method = request.method
            path = request.path
            user_agent = request.META.get('HTTP_USER_AGENT', '')
            referer = request.META.get('HTTP_REFERER', '')

            # Collect all request data
            get_data = dict(request.GET)
            post_data = dict(request.POST) if method == 'POST' else {}

            # Get all headers for header injection detection
            all_headers = ' '.join([f"{k}:{v}" for k, v in request.META.items() if k.startswith('HTTP_')])

            # Combine all data for attack detection
            combined_data = f"{path} {str(get_data)} {str(post_data)} {user_agent} {all_headers}"

            # Detect attack
            attack_type, severity, payload = self.detect_attack(combined_data)

            # Collect headers (sanitize sensitive ones)
            headers = {
                'User-Agent': user_agent,
                'Referer': referer,
                'Content-Type': request.META.get('CONTENT_TYPE', ''),
                'Accept': request.META.get('HTTP_ACCEPT', ''),
                'Host': request.META.get('HTTP_HOST', ''),
            }

            # Create attack log entry
            attack_log = AttackLog(
                ip_address=ip,
                method=method,
                path=path,
                user_agent=user_agent,
                referer=referer,
                attack_type=attack_type or '',
                severity=severity,
                payload=payload or '',
            )
            attack_log.set_get_params(get_data)
            attack_log.set_post_params(post_data)
            attack_log.set_headers(headers)

            # Store in request for later update
            request.attack_log = attack_log

            # Log to file if attack detected
            if attack_type:
                logger.warning(
                    f"ATTACK DETECTED | IP: {ip} | Type: {attack_type} | "
                    f"Severity: {severity} | Path: {path} | Payload: {payload}"
                )

        except Exception as e:
            logger.error(f"Error in AttackLoggerMiddleware.process_request: {e}")

        return None

    def process_response(self, request, response):
        """Log response after processing"""
        try:
            if hasattr(request, 'attack_log'):
                request.attack_log.status_code = response.status_code
                request.attack_log.response_size = len(response.content) if hasattr(response, 'content') else 0
                request.attack_log.save()
        except Exception as e:
            logger.error(f"Error in AttackLoggerMiddleware.process_response: {e}")

        return response
