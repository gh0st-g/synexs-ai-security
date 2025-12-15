#!/usr/bin/env python3
"""
wordpress_upload.py - Upload Synexs documentation to WordPress
Converts Markdown to HTML and posts via WordPress REST API
"""

import requests
import base64
import json
import markdown
from pathlib import Path
from typing import Dict, Union
from datetime import datetime
import logging

# WordPress Configuration
WP_SITE = "https://synexs.net"
WP_API = f"{WP_SITE}/wp-json/wp/v2"
WP_USERNAME = "YOUR_USERNAME"  # Replace with your WordPress username
WP_APP_PASSWORD = "YOUR_APP_PASSWORD"  # Generate this in WordPress

# Documents to upload
DOCS = [
    {
        "file": "SYNEXS_MASTER_DOCUMENTATION.md",
        "title": "Synexs - Complete Project Documentation",
        "slug": "synexs-documentation",
        "category": "documentation"
    },
    {
        "file": "GPU_SETUP_README.md",
        "title": "Synexs GPU Training - Setup Guide",
        "slug": "gpu-setup-guide",
        "category": "documentation"
    },
    {
        "file": "BINARY_PROTOCOL_DEPLOYMENT.md",
        "title": "Binary Protocol V3 - Deployment Guide",
        "slug": "binary-protocol-deployment",
        "category": "documentation"
    },
    {
        "file": "TRANSFER_READY.txt",
        "title": "Synexs - Production Ready Status",
        "slug": "transfer-ready-status",
        "category": "documentation"
    }
]

class WordPressUploader:
    def __init__(self, site_url: str, username: str, app_password: str):
        self.api_url = f"{site_url}/wp-json/wp/v2"
        self.auth = (username, app_password)
        self.headers = {
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('wordpress_upload.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def markdown_to_html(self, md_content: str) -> str:
        """Convert Markdown to HTML"""
        # Configure markdown extensions
        extensions = [
            'markdown.extensions.fenced_code',
            'markdown.extensions.tables',
            'markdown.extensions.toc',
            'markdown.extensions.nl2br',
            'markdown.extensions.sane_lists'
        ]

        html = markdown.markdown(md_content, extensions=extensions)
        return html

    def create_post(self, title: str, content: str, slug: str, status: str = "draft") -> Dict[str, Union[bool, int, str]]:
        """Create a WordPress post"""
        post_data = {
            "title": title,
            "content": content,
            "slug": slug,
            "status": status,  # 'draft' or 'publish'
            "format": "standard"
        }

        try:
            response = requests.post(
                f"{self.api_url}/posts",
                auth=self.auth,
                headers=self.headers,
                json=post_data,
                timeout=60
            )
            response.raise_for_status()
            return {"success": True, "id": response.json()['id'], "url": response.json()['link']}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error creating post: {e}")
            return {"success": False, "error": str(e)}

    def create_page(self, title: str, content: str, slug: str, status: str = "draft") -> Dict[str, Union[bool, int, str]]:
        """Create a WordPress page (better for documentation)"""
        page_data = {
            "title": title,
            "content": content,
            "slug": slug,
            "status": status
        }

        try:
            response = requests.post(
                f"{self.api_url}/pages",
                auth=self.auth,
                headers=self.headers,
                json=page_data,
                timeout=60
            )
            response.raise_for_status()
            return {"success": True, "id": response.json()['id'], "url": response.json()['link']}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error creating page: {e}")
            return {"success": False, "error": str(e)}

    def upload_document(self, file_path: str, title: str, slug: str, as_page: bool = True) -> Dict[str, Union[bool, int, str]]:
        """Upload a markdown document to WordPress"""
        # Read markdown file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return {"success": False, "error": "File not found"}
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return {"success": False, "error": str(e)}

        # Convert to HTML
        html_content = self.markdown_to_html(md_content)

        # Add custom styling
        styled_content = f"""
        <div class="synexs-documentation">
            <style>
                .synexs-documentation {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
                .synexs-documentation pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .synexs-documentation code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}
                .synexs-documentation table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .synexs-documentation th, .synexs-documentation td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .synexs-documentation th {{ background-color: #4CAF50; color: white; }}
                .synexs-documentation h2 {{ border-bottom: 2px solid #4CAF50; padding-bottom: 10px; margin-top: 30px; }}
            </style>
            {html_content}
        </div>
        """

        # Upload as page or post
        if as_page:
            return self.create_page(title, styled_content, slug, status="draft")
        else:
            return self.create_post(title, styled_content, slug, status="draft")

def main():
    print("🌐 Synexs WordPress Documentation Uploader")
    print("=" * 60)

    # Initialize uploader
    uploader = WordPressUploader(WP_SITE, WP_USERNAME, WP_APP_PASSWORD)

    # Upload each document
    results = []
    for doc in DOCS:
        file_path = doc["file"]

        print(f"\n📄 Uploading: {doc['title']}")
        print(f"   File: {file_path}")

        result = uploader.upload_document(
            file_path=file_path,
            title=doc["title"],
            slug=doc["slug"],
            as_page=True  # Use pages for documentation
        )

        if result["success"]:
            print(f"✅ Success! Page ID: {result['id']}")
            print(f"   URL: {result['url']}")
            results.append({
                "file": file_path,
                "status": "success",
                "url": result["url"]
            })
        else:
            print(f"❌ Failed: {result['error']}")
            results.append({
                "file": file_path,
                "status": "failed",
                "error": result["error"]
            })

    # Summary
    print("\n" + "=" * 60)
    print("📊 Upload Summary:")
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"   Successful: {success_count}/{len(results)}")

    if success_count > 0:
        print("\n✅ Uploaded Pages:")
        for r in results:
            if r["status"] == "success":
                print(f"   • {r['url']}")

    # Save results
    try:
        with open("wordpress_upload_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: wordpress_upload_results.json")
    except Exception as e:
        uploader.logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()