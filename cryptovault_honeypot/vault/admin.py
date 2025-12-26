"""
Django Admin configuration
"""
from django.contrib import admin
from .models import User, Wallet, Comment, FileUpload, AttackLog


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'wallet_address', 'balance_btc', 'balance_eth', 'is_premium']
    search_fields = ['username', 'email', 'wallet_address']


@admin.register(Wallet)
class WalletAdmin(admin.ModelAdmin):
    list_display = ['user', 'transaction_type', 'currency', 'amount', 'status', 'timestamp']
    list_filter = ['transaction_type', 'currency', 'status']
    search_fields = ['user__username', 'address_to', 'address_from']


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['user', 'content', 'timestamp']
    search_fields = ['user__username', 'content']


@admin.register(FileUpload)
class FileUploadAdmin(admin.ModelAdmin):
    list_display = ['user', 'filename', 'uploaded_at']
    search_fields = ['user__username', 'filename']


@admin.register(AttackLog)
class AttackLogAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'ip_address', 'method', 'path', 'attack_type', 'severity', 'status_code']
    list_filter = ['attack_type', 'severity', 'method']
    search_fields = ['ip_address', 'path', 'attack_type']
    date_hierarchy = 'timestamp'
    readonly_fields = ['timestamp']

    def has_add_permission(self, request):
        return False  # Logs are auto-generated

    def has_delete_permission(self, request, obj=None):
        return True  # Allow cleanup
