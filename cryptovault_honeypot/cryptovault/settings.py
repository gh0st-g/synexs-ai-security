"""
Django settings for CryptoVault Honeypot
SECURITY WARNING: This is intentionally insecure for honeypot purposes!
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# VULN: Exposed secret key in settings
SECRET_KEY = 'django-insecure-cryptovault-honeypot-DO-NOT-USE-IN-PROD-123456789'

# VULN: Debug mode enabled in production
DEBUG = True

# VULN: Allow all hosts
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'graphene_django',
    'vault',
]

# Custom user model
AUTH_USER_MODEL = 'vault.User'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # VULN: CSRF middleware disabled globally
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    # VULN: XSS protection disabled
    # 'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'vault.middleware.AttackLoggerMiddleware',  # Our custom logging
]

ROOT_URLCONF = 'cryptovault.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            # VULN: Allow string_if_invalid for SSTI testing
            'string_if_invalid': 'INVALID_VARIABLE',
        },
    },
    # VULN: Add Jinja2 backend for SSTI vulnerabilities
    {
        'BACKEND': 'django.template.backends.jinja2.Jinja2',
        'DIRS': [BASE_DIR / 'templates' / 'jinja2'],
        'APP_DIRS': True,
        'OPTIONS': {
            'environment': 'vault.jinja2.environment',
        },
    },
]

WSGI_APPLICATION = 'cryptovault.wsgi.application'

# Database - SQLite for honeypot logs and attack data
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# MongoDB for NoSQL injection vulnerability
MONGODB_SETTINGS = {
    'host': 'localhost',
    'port': 27017,
    'database': 'cryptovault_nosql',
}

# LDAP settings (dummy for honeypot)
LDAP_SERVER = 'ldap://localhost:389'
LDAP_BASE_DN = 'dc=cryptovault,dc=com'

# Password validation - VULN: Weak/disabled
AUTH_PASSWORD_VALIDATORS = []

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Media files - VULN: Unrestricted upload
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/root/synexs/cryptovault_honeypot/logs/honeypot.log',
            'formatter': 'verbose',
        },
        'attack_file': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': '/root/synexs/cryptovault_honeypot/logs/attacks.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'vault.attacks': {
            'handlers': ['attack_file'],
            'level': 'WARNING',
            'propagate': False,
        },
        'vault': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# GraphQL
GRAPHENE = {
    'SCHEMA': 'vault.schema.schema',
    # VULN: Introspection enabled
    'MIDDLEWARE': [],
}

# VULN: Security settings disabled
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SECURE_BROWSER_XSS_FILTER = False
SECURE_CONTENT_TYPE_NOSNIFF = False
X_FRAME_OPTIONS = 'ALLOW'
