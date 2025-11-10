#!/usr/bin/env python3
"""
convert_docs_to_html.py - Convert Markdown docs to HTML for WordPress
Generates HTML files you can copy-paste into WordPress editor
"""

import markdown
import os
from pathlib import Path

# Documents to convert
DOCS = [
    "SYNEXS_MASTER_DOCUMENTATION.md",
    "GPU_SETUP_README.md",
    "BINARY_PROTOCOL_DEPLOYMENT.md",
    "BINARY_PROTOCOL_COMPLETE.md",
    "TRANSFER_READY.txt"
]

OUTPUT_DIR = "wordpress_html"

def markdown_to_html(md_file):
    """Convert markdown file to styled HTML"""

    # Read file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Configure markdown extensions
    extensions = [
        'markdown.extensions.fenced_code',
        'markdown.extensions.tables',
        'markdown.extensions.toc',
        'markdown.extensions.nl2br',
        'markdown.extensions.sane_lists',
        'markdown.extensions.codehilite'
    ]

    # Convert to HTML
    html_body = markdown.markdown(content, extensions=extensions)

    # Create full HTML with styling
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{Path(md_file).stem}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}

        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }}

        h3 {{
            color: #555;
            margin-top: 20px;
        }}

        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', Consolas, Monaco, monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }}

        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }}

        pre code {{
            background: transparent;
            padding: 0;
            color: #ecf0f1;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            border: 1px solid #ddd;
            padding: 12px;
        }}

        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
            background: #f9f9f9;
            padding: 10px 20px;
        }}

        a {{
            color: #3498db;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        ul, ol {{
            padding-left: 25px;
        }}

        li {{
            margin: 8px 0;
        }}

        .emoji {{
            font-size: 1.2em;
        }}

        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}

        .highlight {{
            background: #fffbcc;
            padding: 2px 4px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
    """

    return html

def main():
    print("📝 Converting Synexs Documentation to HTML")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    converted = []

    for doc in DOCS:
        if not os.path.exists(doc):
            print(f"⚠️  File not found: {doc}")
            continue

        print(f"\n📄 Converting: {doc}")

        # Convert to HTML
        html = markdown_to_html(doc)

        # Output filename
        output_file = os.path.join(OUTPUT_DIR, Path(doc).stem + ".html")

        # Write HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✅ Saved to: {output_file}")
        converted.append(output_file)

    # Summary
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
