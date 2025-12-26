"""
Views for CryptoVault Honeypot
WARNING: All vulnerabilities are INTENTIONAL for honeypot purposes
"""
import os
import sqlite3
import json
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import requests
from pymongo import MongoClient

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db import connection

from .models import User, Wallet, Comment, FileUpload, AttackLog
from jinja2 import Template


# ==================== Homepage ====================

def home(request):
    """Homepage"""
    return render(request, 'vault/home.html')


# ==================== Authentication ====================

@csrf_exempt  # VULN: No CSRF protection
def login_view(request):
    """
    Login page
    VULN: SQL Injection in authentication
    """
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')

        # VULN: Raw SQL with string formatting - SQL Injection
        query = f"SELECT * FROM vault_users WHERE username = '{username}' AND password = '{password}'"

        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                user_data = cursor.fetchone()

                if user_data:
                    # Authenticate user
                    user = authenticate(request, username=username, password=password)
                    if user:
                        login(request, user)
                        return redirect('dashboard')
                    else:
                        # Try to get user by username for demo
                        try:
                            user = User.objects.get(username=username)
                            login(request, user)
                            return redirect('dashboard')
                        except User.DoesNotExist:
                            error = "Invalid credentials"
                else:
                    error = "Invalid credentials"
        except Exception as e:
            # VULN: Expose SQL errors
            error = f"Database error: {str(e)}"

    return render(request, 'vault/login.html', {'error': error})


@csrf_exempt  # VULN: No CSRF protection
def register_view(request):
    """
    Registration page - REAL SIGNUP!
    VULN: No input validation, weak passwords allowed
    """
    import random
    from decimal import Decimal

    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '')
        email = request.POST.get('email', '')
        password = request.POST.get('password', '')  # VULN: No hashing demo

        try:
            # Create user
            user = User.objects.create_user(username=username, email=email, password=password)

            # Generate realistic wallet data
            user.wallet_address = f"0x{os.urandom(20).hex()}"

            # Generate random seed phrase (12 words from BIP39 word list sample)
            words = ['abandon', 'ability', 'able', 'about', 'above', 'absent', 'absorb', 'abstract',
                     'absurd', 'abuse', 'access', 'accident', 'account', 'accuse', 'achieve', 'acid',
                     'acoustic', 'acquire', 'across', 'act', 'action', 'actor', 'actress', 'actual',
                     'adapt', 'add', 'addict', 'address', 'adjust', 'admit', 'adult', 'advance',
                     'advice', 'aerobic', 'affair', 'afford', 'afraid', 'again', 'age', 'agent',
                     'agree', 'ahead', 'aim', 'air', 'airport', 'aisle', 'alarm', 'album',
                     'horse', 'battery', 'staple', 'correct', 'wallet', 'crypto', 'secure', 'vault',
                     'invest', 'bitcoin', 'ethereum', 'digital', 'decentralized', 'blockchain']
            user.wallet_seed = ' '.join(random.sample(words, 12))

            # Random realistic balances
            user.balance_btc = Decimal(str(round(random.uniform(0.001, 5.0), 8)))
            user.balance_eth = Decimal(str(round(random.uniform(0.1, 20.0), 6)))
            user.balance_usdt = Decimal(str(round(random.uniform(100, 50000), 2)))

            # Random premium status (30% chance)
            user.is_premium = random.random() < 0.3

            # Generate API key
            user.api_key = f"sk_{os.urandom(16).hex()}"

            user.save()

            # Create initial transactions
            from .models import Wallet
            tx_types = ['deposit', 'trade', 'withdraw']
            currencies = ['BTC', 'ETH', 'USDT']

            for _ in range(random.randint(2, 5)):
                Wallet.objects.create(
                    user=user,
                    transaction_type=random.choice(tx_types),
                    currency=random.choice(currencies),
                    amount=Decimal(str(round(random.uniform(0.001, 2.0), 8))),
                    address_to=f"0x{os.urandom(20).hex()}" if random.choice([True, False]) else '',
                    status='completed',
                )

            login(request, user)
            return redirect('dashboard')
        except Exception as e:
            error = str(e)

    return render(request, 'vault/register.html', {'error': error})


def logout_view(request):
    """
    Logout
    VULN: Open Redirect
    """
    logout(request)
    # VULN: No validation on redirect parameter
    next_url = request.GET.get('next', '/')
    return redirect(next_url)


# ==================== Dashboard ====================

@login_required
def dashboard(request):
    """
    User dashboard
    VULN: IDOR - Can access other users' data via ?user_id=
    """
    # VULN: Insecure Direct Object Reference
    user_id = request.GET.get('user_id', request.user.id)

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        user = request.user

    transactions = Wallet.objects.filter(user=user)[:10]

    return render(request, 'vault/dashboard.html', {
        'user': user,
        'transactions': transactions,
    })


# ==================== Search ====================

@login_required
def search(request):
    """
    Search transactions
    VULN: SQL Injection in search query
    """
    query = request.GET.get('q', '')
    results = []
    error = None

    if query:
        # VULN: Raw SQL injection vulnerability
        sql = f"SELECT * FROM vault_wallets WHERE transaction_type LIKE '%{query}%' OR currency LIKE '%{query}%'"

        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                columns = [col[0] for col in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            # VULN: Expose database errors
            error = f"SQL Error: {str(e)}"

    return render(request, 'vault/search.html', {
        'query': query,
        'results': results,
        'error': error,
    })


# ==================== Comments ====================

@csrf_exempt  # VULN: No CSRF
@login_required
def comments(request):
    """
    Comments page
    VULN: Reflected XSS in comment display
    """
    if request.method == 'POST':
        content = request.POST.get('comment', '')
        # VULN: No sanitization - stored XSS
        Comment.objects.create(user=request.user, content=content)

    # VULN: Reflected XSS via GET parameter
    highlight = request.GET.get('highlight', '')

    all_comments = Comment.objects.all()[:20]

    return render(request, 'vault/comments.html', {
        'comments': all_comments,
        'highlight': highlight,  # VULN: Rendered without escaping
    })


# ==================== File Operations ====================

@csrf_exempt
@login_required
def download_file(request):
    """
    Download files
    VULN: Path Traversal vulnerability
    """
    # VULN: No path validation - allows directory traversal
    filename = request.GET.get('file', '')

    if not filename:
        return HttpResponse("No file specified", status=400)

    # VULN: Direct file access without validation
    filepath = os.path.join(settings.BASE_DIR, filename)

    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                response = HttpResponse(f.read())
                response['Content-Disposition'] = f'attachment; filename="{os.path.basename(filename)}"'
                return response
        else:
            return HttpResponse(f"File not found: {filepath}", status=404)
    except Exception as e:
        # VULN: Expose system errors
        return HttpResponse(f"Error: {str(e)}", status=500)


@csrf_exempt
@login_required
def upload_file(request):
    """
    File upload
    VULN: Unrestricted file upload - allows webshells
    """
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # VULN: No file type validation, no size limit, no content check
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        filepath = os.path.join(upload_dir, uploaded_file.name)

        with open(filepath, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        FileUpload.objects.create(
            user=request.user,
            filename=uploaded_file.name,
            filepath=filepath,
            file=uploaded_file,
        )

        return JsonResponse({
            'success': True,
            'filename': uploaded_file.name,
            'path': filepath,
        })

    return render(request, 'vault/upload.html')


# ==================== System Check ====================

@csrf_exempt
def system_check(request):
    """
    System health check endpoint
    VULN: Command Injection
    """
    # VULN: Execute user input directly
    command = request.GET.get('cmd', 'whoami')

    try:
        # VULN: No validation - allows arbitrary command execution
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, timeout=5)
        output = result.decode('utf-8')
    except subprocess.TimeoutExpired:
        output = "Command timeout"
    except Exception as e:
        output = f"Error: {str(e)}"

    return JsonResponse({
        'command': command,
        'output': output,
    })


# ==================== Price API ====================

@csrf_exempt
def fetch_price(request):
    """
    Fetch cryptocurrency price from external API
    VULN: Server-Side Request Forgery (SSRF)
    """
    # VULN: No URL validation - allows SSRF
    url = request.GET.get('url', 'https://api.coinbase.com/v2/prices/BTC-USD/spot')

    try:
        # VULN: Fetches any URL without validation
        response = requests.get(url, timeout=5)
        data = response.text

        return JsonResponse({
            'url': url,
            'status_code': response.status_code,
            'data': data[:1000],  # Limit response size
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'url': url,
        })


# ==================== XML Upload ====================

@csrf_exempt
def xml_upload(request):
    """
    Upload and parse XML
    VULN: XML External Entity (XXE) Injection
    """
    if request.method == 'POST':
        xml_data = request.POST.get('xml', '')

        try:
            # VULN: Unsafe XML parsing - allows XXE
            parser = ET.XMLParser()  # No defusedxml, vulnerable to XXE
            root = ET.fromstring(xml_data, parser=parser)

            # Extract data
            result = {
                'tag': root.tag,
                'attrib': root.attrib,
                'text': root.text,
                'children': [child.tag for child in root],
            }

            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)})

    return render(request, 'vault/xml_upload.html')


# ==================== Template Rendering ====================

@csrf_exempt
def render_template(request):
    """
    Custom template renderer
    VULN: Server-Side Template Injection (SSTI)
    """
    # VULN: User-controlled template content
    template_str = request.GET.get('template', 'Hello {{name}}!')
    name = request.GET.get('name', 'User')

    try:
        # VULN: Rendering untrusted template with Jinja2
        template = Template(template_str)
        rendered = template.render(name=name, user=request.user)

        return HttpResponse(rendered)
    except Exception as e:
        return HttpResponse(f"Template error: {str(e)}")


# ==================== NoSQL Operations ====================

@csrf_exempt
def nosql_search(request):
    """
    Search users in MongoDB
    VULN: NoSQL Injection
    """
    username = request.GET.get('username', '')

    try:
        # Connect to MongoDB
        client = MongoClient(
            settings.MONGODB_SETTINGS['host'],
            settings.MONGODB_SETTINGS['port']
        )
        db = client[settings.MONGODB_SETTINGS['database']]

        # VULN: Unsanitized user input in MongoDB query
        # Example attack: ?username[$ne]=null
        if username.startswith('{'):
            # VULN: Allows JSON injection
            query = json.loads(username)
        else:
            query = {'username': username}

        results = list(db.users.find(query).limit(10))

        # Convert ObjectId to string
        for r in results:
            r['_id'] = str(r['_id'])

        return JsonResponse({'results': results})
    except Exception as e:
        return JsonResponse({'error': str(e)})


# ==================== LDAP Search ====================

@csrf_exempt
def ldap_search(request):
    """
    LDAP user search (simulated)
    VULN: LDAP Injection
    """
    username = request.GET.get('username', '')

    # VULN: Unsanitized LDAP filter
    # In real LDAP, this would be: f"(uid={username})"
    # Attack: username=*)(uid=*))(|(uid=*

    ldap_filter = f"(uid={username})"

    # Simulate LDAP response (not real LDAP for demo)
    results = {
        'filter': ldap_filter,
        'message': 'LDAP search simulated (vulnerable to injection)',
        'sample_attack': "Try: ?username=*)(uid=*))(|(uid=*",
    }

    return JsonResponse(results)


# ==================== Admin Panel ====================

@csrf_exempt
def admin_panel(request):
    """
    Admin panel
    VULN: Broken Access Control - no admin check
    VULN: HTTP Parameter Pollution in role check
    """
    # VULN: Checks GET parameter instead of actual user role
    role = request.GET.get('role', 'user')

    # VULN: Can be bypassed with ?role=admin or HPP: ?role=user&role=admin
    is_admin = role == 'admin'

    users = User.objects.all() if is_admin else []
    attack_logs = AttackLog.objects.all()[:50] if is_admin else []

    return render(request, 'admin_panel/admin.html', {
        'is_admin': is_admin,
        'users': users,
        'attack_logs': attack_logs,
    })


# ==================== Password Reset ====================

@csrf_exempt
def reset_password(request):
    """
    Password reset
    VULN: Header Injection in reset link
    """
    if request.method == 'POST':
        email = request.POST.get('email', '')

        # VULN: User input in headers without validation
        reset_link = f"https://cryptovault.com/reset?email={email}"

        # Simulate email (log it instead)
        response = HttpResponse("Password reset email sent!")

        # VULN: Header injection - allows response splitting
        response['X-Reset-Link'] = reset_link
        response['X-User-Email'] = email  # VULN: Unsanitized

        return response

    return render(request, 'vault/reset_password.html')


# ==================== API Endpoints ====================

@csrf_exempt
def api_wallets(request):
    """
    REST API - Get all wallets
    VULN: No authentication, exposes all data
    """
    # VULN: No auth required, returns sensitive data
    wallets = Wallet.objects.all()[:100]

    data = [{
        'id': w.id,
        'user': w.user.username,
        'type': w.transaction_type,
        'amount': float(w.amount),
        'currency': w.currency,
        'timestamp': w.timestamp.isoformat(),
    } for w in wallets]

    return JsonResponse({'wallets': data})


@csrf_exempt
def api_user_info(request):
    """
    User info API
    VULN: IDOR - access any user's info
    """
    user_id = request.GET.get('id', 1)

    try:
        user = User.objects.get(id=user_id)
        data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'wallet_address': user.wallet_address,
            'wallet_seed': user.wallet_seed,  # VULN: Exposing seed phrase!
            'balance_btc': float(user.balance_btc),
            'balance_eth': float(user.balance_eth),
            'balance_usdt': float(user.balance_usdt),
        }
        return JsonResponse(data)
    except User.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)


# ==================== Debug Info ====================

def debug_info(request):
    """
    Debug information page
    VULN: Security Misconfiguration - exposes sensitive info
    """
    info = {
        'django_version': '5.0.1',
        'debug_mode': settings.DEBUG,
        'secret_key': settings.SECRET_KEY[:20] + '...',  # VULN: Partial key leak
        'database': settings.DATABASES['default']['NAME'],
        'allowed_hosts': settings.ALLOWED_HOSTS,
        'installed_apps': settings.INSTALLED_APPS,
        'middleware': settings.MIDDLEWARE,
        'python_path': os.sys.path,
        'environment_vars': dict(os.environ),  # VULN: Full env exposure
    }

    return JsonResponse(info, json_dumps_params={'indent': 2})
