#!/usr/bin/env python3
"""
convert_docs_to_html.py - Convert Markdown docs to HTML for WordPress
Generates HTML files you can copy-paste into WordPress editor
"""

import markdown
import os
from pathlib import Path
from typing import List

# Documents to convert
DOCS: List[str] = [
    "SYNEXS_MASTER_DOCUMENTATION.md",
    "GPU_SETUP_README.md",
    "BINARY_PROTOCOL_DEPLOYMENT.md",
    "BINARY_PROTOCOL_COMPLETE.md",
    "TRANSFER_READY.txt"
]

OUTPUT_DIR: str = "wordpress_html"

def markdown_to_html(md_file: str) -> str:
    """Convert markdown file to styled HTML"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"⚠️  File not found: {md_file}")
        return ""

    extensions = [
        'markdown.extensions.fenced_code',
        'markdown.extensions.tables',
        'markdown.extensions.toc',
        'markdown.extensions.nl2br',
        'markdown.extensions.sane_lists',
        'markdown.extensions.codehilite'
    ]

    html_body = markdown.markdown(content, extensions=extensions)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{Path(md_file).stem}</title>
    <style>
        /* Styling omitted for brevity */
    </style>
</head>
<body>
{html_body}
</body>
</html>
    """

    return html

def main() -> None:
    print("📝 Converting Synexs Documentation to HTML")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    converted: List[str] = []

    for doc in DOCS:
        print(f"\n📄 Converting: {doc}")

        try:
            html = markdown_to_html(doc)
            if html:
                output_file = os.path.join(OUTPUT_DIR, Path(doc).stem + ".html")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"✅ Saved to: {output_file}")
                converted.append(output_file)
        except Exception as e:
            print(f"⚠️  Error converting {doc}: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"✅ Converted {len(converted)} documents")
    print(f"📂 Output directory: {OUTPUT_DIR}/")
    print("\n📋 Next Steps:")
    print("   1. Open each HTML file in your browser")
    print("   2. Copy the content")
    print("   3. In WordPress, create new Page")
    print("   4. Switch to 'Custom HTML' block")
    print("   5. Paste the HTML content")
    print("   6. Publish!")

if __name__ == "__main__":
    main()