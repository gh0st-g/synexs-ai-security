"""
Models for CryptoVault Honeypot
"""
from django.db import models
from django.contrib.auth.models import AbstractUser
import json


class User(AbstractUser):
    """
    Extended user model with wallet data
    """
    wallet_address = models.CharField(max_length=100, blank=True)
    wallet_seed = models.CharField(max_length=500, blank=True)  # VULN: Storing seed phrase!
    balance_btc = models.DecimalField(max_digits=20, decimal_places=8, default=0)
    balance_eth = models.DecimalField(max_digits=20, decimal_places=8, default=0)
    balance_usdt = models.DecimalField(max_digits=20, decimal_places=2, default=0)
    api_key = models.CharField(max_length=100, blank=True)
    is_premium = models.BooleanField(default=False)

    # Fix for Django 4.2+ - avoid reverse accessor clashes
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to.',
        related_name='vault_users',
        related_query_name='vault_user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='vault_users',
        related_query_name='vault_user',
    )

    class Meta:
        db_table = 'vault_users'

    def __str__(self):
        return self.username


class Wallet(models.Model):
    """
    Wallet transactions
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transactions')
    transaction_type = models.CharField(max_length=20)  # deposit, withdraw, trade
    currency = models.CharField(max_length=10)
    amount = models.DecimalField(max_digits=20, decimal_places=8)
    address_to = models.CharField(max_length=100, blank=True)
    address_from = models.CharField(max_length=100, blank=True)
    status = models.CharField(max_length=20, default='completed')
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'vault_wallets'
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username} - {self.transaction_type} - {self.amount} {self.currency}"


class AttackLog(models.Model):
    """
    Log all HTTP requests and potential attacks
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField()
    method = models.CharField(max_length=10)
    path = models.CharField(max_length=500)
    user_agent = models.TextField(blank=True)
    referer = models.TextField(blank=True)

    # Request data
    get_params = models.TextField(blank=True)  # JSON
    post_params = models.TextField(blank=True)  # JSON
    headers = models.TextField(blank=True)  # JSON

    # Attack detection
    attack_type = models.CharField(max_length=100, blank=True)  # sqli, xss, lfi, etc.
    severity = models.CharField(max_length=20, default='low')  # low, medium, high, critical
    payload = models.TextField(blank=True)

    # Response
    status_code = models.IntegerField(null=True)
    response_size = models.IntegerField(null=True)

    class Meta:
        db_table = 'attack_logs'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['ip_address', 'timestamp']),
            models.Index(fields=['attack_type']),
            models.Index(fields=['severity']),
        ]

    def __str__(self):
        return f"{self.ip_address} - {self.method} {self.path} - {self.attack_type or 'normal'}"

    def set_get_params(self, params_dict):
        self.get_params = json.dumps(params_dict)

    def set_post_params(self, params_dict):
        self.post_params = json.dumps(params_dict)

    def set_headers(self, headers_dict):
        self.headers = json.dumps(headers_dict)


class Comment(models.Model):
    """
    User comments - vulnerable to XSS
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()  # VULN: Not sanitized
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'vault_comments'
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username}: {self.content[:50]}"


class FileUpload(models.Model):
    """
    File uploads - unrestricted
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    filepath = models.CharField(max_length=500)
    file = models.FileField(upload_to='uploads/')  # VULN: No validation
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'vault_uploads'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.user.username} - {self.filename}"
