"""
URL Configuration for Vault app
"""
from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),

    # Vulnerable endpoints
    path('search/', views.search, name='search'),  # SQLi
    path('comments/', views.comments, name='comments'),  # XSS
    path('download/', views.download_file, name='download'),  # Path Traversal
    path('upload/', views.upload_file, name='upload'),  # Unrestricted upload
    path('system-check/', views.system_check, name='system_check'),  # RCE
    path('fetch-price/', views.fetch_price, name='fetch_price'),  # SSRF
    path('xml-upload/', views.xml_upload, name='xml_upload'),  # XXE
    path('render/', views.render_template, name='render_template'),  # SSTI
    path('nosql-search/', views.nosql_search, name='nosql_search'),  # NoSQL injection
    path('ldap-search/', views.ldap_search, name='ldap_search'),  # LDAP injection
    path('reset-password/', views.reset_password, name='reset_password'),  # Header injection

    # Admin panel
    path('admin-panel/', views.admin_panel, name='admin_panel'),  # Broken access control

    # API endpoints
    path('api/wallets/', views.api_wallets, name='api_wallets'),
    path('api/user/', views.api_user_info, name='api_user_info'),  # IDOR

    # Debug
    path('debug/', views.debug_info, name='debug_info'),  # Security misconfiguration
]
