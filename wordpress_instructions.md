# Adding Synexs Documentation to WordPress

There are **4 methods** to add your documentation to synexs.net:

---

## **Method 1: WordPress REST API (Automated)** ⚡

### Prerequisites:
1. Generate WordPress Application Password:
   - Go to: https://synexs.net/wp-admin/profile.php
   - Scroll to "Application Passwords"
   - Enter name: "Synexs API Upload"
   - Click "Add New Application Password"
   - **Copy the password** (shows only once!)

2. Install Python dependencies:
```bash
pip3 install markdown requests
```

3. Edit `wordpress_upload.py`:
```bash
nano wordpress_upload.py
```

Change these lines:
```python
WP_USERNAME = "your_actual_username"  # Your WordPress username
WP_APP_PASSWORD = "xxxx xxxx xxxx xxxx"  # The app password you generated
```

4. Run the uploader:
```bash
python3 wordpress_upload.py
```

**Result**: Creates WordPress pages automatically (as drafts for you to review)

---

## **Method 2: Convert to HTML (Copy-Paste)** 📋

### Steps:

1. Install markdown library:
```bash
pip3 install markdown
```

2. Run the converter:
```bash
python3 convert_docs_to_html.py
```

3. Output files will be in `wordpress_html/` folder

4. For each HTML file:
   - Open in browser or text editor
   - Copy entire content
   - Go to WordPress: Pages → Add New
   - Switch to "Custom HTML" block
   - Paste the HTML
   - Preview and Publish

**Advantage**: Full control, no API needed

---

## **Method 3: Use WordPress Plugin** 🔌

### Option A: WP Githuber MD Plugin

1. Install plugin:
   - Go to: https://synexs.net/wp-admin/plugin-install.php
   - Search: "WP Githuber MD"
   - Install and Activate

2. Upload documents:
   - Pages → Add New
   - Switch to "Markdown" mode
   - Copy-paste markdown content directly
   - Publish

### Option B: Markdown Block Plugin

1. Install "Markdown Block for Gutenberg"
2. Create new page
3. Add "Markdown" block
4. Paste markdown content
5. Publish

**Advantage**: Native WordPress markdown support

---

## **Method 4: Direct FTP/SFTP Upload** 📁

### Steps:

1. Convert to HTML first:
```bash
python3 convert_docs_to_html.py
```

2. Upload HTML files via FTP/SFTP to:
```
/var/www/html/synexs.net/wp-content/uploads/docs/
```

3. Create WordPress pages with links:
```html
<iframe src="/wp-content/uploads/docs/SYNEXS_MASTER_DOCUMENTATION.html"
        width="100%"
        height="800px"
        style="border: none;">
</iframe>
```

**Advantage**: Files hosted on your server, easy to update

---

## **Recommended Approach**

### **For Quick Setup: Method 2 (HTML Copy-Paste)**

1. Run converter:
```bash
python3 convert_docs_to_html.py
```

2. Open `wordpress_html/SYNEXS_MASTER_DOCUMENTATION.html` in browser

3. Copy all content (Ctrl+A, Ctrl+C)

4. WordPress: Pages → Add New
   - Title: "Synexs Documentation"
   - Slug: "documentation"
   - Add "Custom HTML" block
   - Paste content
   - Publish

5. Repeat for other docs:
   - GPU Setup Guide → `/gpu-setup`
   - Binary Protocol → `/binary-protocol`
   - Transfer Ready → `/status`

---

## **Creating Documentation Menu**

After uploading, create a menu structure:

1. Go to: Appearance → Menus
2. Create new menu: "Documentation"
3. Add your pages:
   - Synexs Documentation
   - GPU Setup Guide
   - Binary Protocol
   - System Status

4. Set menu location (depends on your theme)

---

## **Alternative: Create Documentation Hub Page**

Create a main page: https://synexs.net/docs

Content:
```html
<h1>Synexs Documentation</h1>

<div class="doc-grid">
  <div class="doc-card">
    <h2>📚 Complete Documentation</h2>
    <p>Full project specification, architecture, and components</p>
    <a href="/documentation">Read More →</a>
  </div>

  <div class="doc-card">
    <h2>🚀 GPU Setup Guide</h2>
    <p>Training setup, chatbot integration, and deployment</p>
    <a href="/gpu-setup">Read More →</a>
  </div>

  <div class="doc-card">
    <h2>⚡ Binary Protocol</h2>
    <p>Protocol V3 specification and implementation</p>
    <a href="/binary-protocol">Read More →</a>
  </div>

  <div class="doc-card">
    <h2>✅ System Status</h2>
    <p>Current status and next steps</p>
    <a href="/status">Read More →</a>
  </div>
</div>

<style>
.doc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}
.doc-card {
  border: 2px solid #3498db;
  padding: 20px;
  border-radius: 8px;
  transition: transform 0.2s;
}
.doc-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.doc-card h2 {
  margin-top: 0;
  color: #2c3e50;
}
.doc-card a {
  color: #3498db;
  text-decoration: none;
  font-weight: bold;
}
</style>
```

---

## **Testing URLs**

After upload, your docs will be at:
- https://synexs.net/documentation/
- https://synexs.net/gpu-setup/
- https://synexs.net/binary-protocol/
- https://synexs.net/status/

---

## **Quick Start (Fastest Method)**

```bash
# 1. Install dependency
pip3 install markdown

# 2. Convert to HTML
python3 convert_docs_to_html.py

# 3. Open in browser
cd wordpress_html
python3 -m http.server 8080

# 4. Visit: http://YOUR_VPS_IP:8080/SYNEXS_MASTER_DOCUMENTATION.html

# 5. Copy-paste into WordPress
```

---

## **Need Help?**

If you run into issues:
1. Check WordPress user permissions (must be Admin/Editor)
2. Verify theme supports Custom HTML blocks
3. Try "Classic Editor" if Gutenberg has issues
4. Contact me for API upload assistance

---

**Recommended: Start with Method 2 (HTML Copy-Paste) - It's the fastest and requires no WordPress configuration!**
