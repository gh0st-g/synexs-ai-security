#!/usr/bin/env python3
"""
Automated WordPress Documentation Uploader
Uploads Synexs HTML documentation files to WordPress via REST API
"""

import requests
import os
import base64
import json
from pathlib import Path

# WordPress Configuration
WP_SITE = "https://synexs.net"
WP_USERNAME = "synexs"
WP_APP_PASSWORD = "xec5 xdlc wakz ryhx"  # Remove spaces for API

# WordPress.com uses public-api.wordpress.com
# Need to determine site ID first
WP_API_URL = f"{WP_SITE}/wp-json/wp/v2"
WPCOM_API_URL = "https://public-api.wordpress.com/wp/v2/sites/synexs.net"

# Remove spaces from app password
WP_APP_PASSWORD_CLEAN = WP_APP_PASSWORD.replace(" ", "")

# Create authentication header
def get_auth_header():
    credentials = f"{WP_USERNAME}:{WP_APP_PASSWORD_CLEAN}"
    token = base64.b64encode(credentials.encode()).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json'
    }

# Test WordPress connection
def test_connection():
    print(f"🔍 Testing connection to {WP_SITE}...")

    # Try WordPress.com API first
    try:
        response = requests.get(f"{WPCOM_API_URL}/posts", headers=get_auth_header(), timeout=10)
        if response.status_code == 200:
            print("✅ Connection successful! (WordPress.com API)")
            return WPCOM_API_URL
        elif response.status_code == 401:
            print("❌ Authentication failed. Please check your credentials.")
            return None
    except Exception as e:
        print(f"⚠️  WordPress.com API error: {e}")

    # Try self-hosted WordPress API
    try:
        response = requests.get(f"{WP_API_URL}/posts", headers=get_auth_header(), timeout=10)
        if response.status_code == 200:
            print("✅ Connection successful! (Self-hosted API)")
            return WP_API_URL
        else:
            print(f"❌ Connection failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

# Read HTML file content
def read_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Extract title from filename
def get_title_from_filename(filename):
    # Remove .html extension and replace underscores with spaces
    title = filename.replace('.html', '').replace('_', ' ')
    # Capitalize words
    return title.title()

# Create WordPress post/page
def create_wordpress_post(title, content, api_url, post_type='post', status='draft'):
    print(f"\n📝 Creating {post_type}: {title}")

    post_data = {
        'title': title,
        'content': content,
        'status': status,  # draft, publish, private
        'format': 'standard'
    }

    endpoint = f"{api_url}/{post_type}s"

    try:
        response = requests.post(
            endpoint,
            headers=get_auth_header(),
            json=post_data,
            timeout=30
        )

        if response.status_code == 201:
            result = response.json()
            post_id = result['id']
            post_url = result['link']
            print(f"✅ Created successfully!")
            print(f"   ID: {post_id}")
            print(f"   URL: {post_url}")
            return {'success': True, 'id': post_id, 'url': post_url}
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {'success': False, 'error': response.text}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

# Main upload function
def upload_documentation(docs_dir, api_url, post_type='post', status='draft'):
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"❌ Directory not found: {docs_dir}")
        return

    html_files = sorted(docs_path.glob('*.html'))

    if not html_files:
        print(f"❌ No HTML files found in {docs_dir}")
        return

    print(f"\n📚 Found {len(html_files)} documentation files:")
    for f in html_files:
        print(f"   - {f.name}")

    print(f"\n🚀 Starting upload process...")
    print(f"   Post type: {post_type}")
    print(f"   Status: {status}")

    results = []

    for html_file in html_files:
        title = get_title_from_filename(html_file.name)
        content = read_html_file(html_file)

        result = create_wordpress_post(title, content, api_url, post_type, status)
        results.append({
            'file': html_file.name,
            'title': title,
            'result': result
        })

    # Summary
    print("\n" + "="*60)
    print("📊 UPLOAD SUMMARY")
    print("="*60)

    successful = [r for r in results if r['result'].get('success')]
    failed = [r for r in results if not r['result'].get('success')]

    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"   • {r['title']}")
        print(f"     {r['result']['url']}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   • {r['title']}")
            print(f"     Error: {r['result'].get('error', 'Unknown')}")

    print("\n" + "="*60)

    return results

if __name__ == "__main__":
    print("="*60)
    print("🌐 SYNEXS WORDPRESS DOCUMENTATION UPLOADER")
    print("="*60)

    # Test connection first
    api_url = test_connection()
    if not api_url:
        print("\n⚠️  Connection test failed. Please check:")
        print("   1. WordPress site URL is correct")
        print("   2. Application password is valid")
        print("   3. WordPress REST API is enabled")
        print("   4. Site is accessible")
        exit(1)

    # Documentation directory
    docs_dir = os.path.expanduser("~/Downloads/wordpress_html")

    # Ask user preferences
    print("\n" + "="*60)
    print("⚙️  UPLOAD CONFIGURATION")
    print("="*60)
    print("\nChoose post type:")
    print("  1. Posts (blog posts, dated content)")
    print("  2. Pages (static pages, no date)")

    post_type_choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    post_type = "page" if post_type_choice == "2" else "post"

    print("\nChoose status:")
    print("  1. Draft (private, not visible)")
    print("  2. Publish (public, immediately visible)")
    print("  3. Private (visible only to admins)")

    status_choice = input("\nEnter choice (1, 2, or 3) [default: 1]: ").strip() or "1"
    status_map = {"1": "draft", "2": "publish", "3": "private"}
    status = status_map.get(status_choice, "draft")

    # Confirm
    print(f"\n📋 Configuration:")
    print(f"   Directory: {docs_dir}")
    print(f"   Post type: {post_type}")
    print(f"   Status: {status}")
    print(f"   Site: {WP_SITE}")

    confirm = input("\n⚠️  Proceed with upload? (yes/no) [default: yes]: ").strip().lower()
    if confirm and confirm not in ['yes', 'y']:
        print("❌ Upload cancelled.")
        exit(0)

    # Upload!
    results = upload_documentation(docs_dir, api_url, post_type, status)

    print("\n✨ Done!")
