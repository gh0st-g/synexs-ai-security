# WordPress Manual Upload Guide for Synexs Documentation

## Issue
The WordPress REST API returned a permissions error. WordPress.com has stricter access controls that prevent automated uploads even with application passwords.

## Solution: Manual Upload via Dashboard

### Files Ready for Upload
Location: `/root/Downloads/wordpress_html/`

| File | Title for WordPress | Size |
|------|---------------------|------|
| TRANSFER_READY.html | Transfer Ready | 12KB |
| BINARY_PROTOCOL_COMPLETE.html | Binary Protocol Complete | 28KB |
| BINARY_PROTOCOL_DEPLOYMENT.html | Binary Protocol Deployment | 35KB |
| GPU_SETUP_README.html | GPU Setup README | 40KB |
| SYNEXS_MASTER_DOCUMENTATION.html | Synexs Master Documentation | 81KB |

### Upload Steps

#### Option A: Direct Copy/Paste (Recommended)

For each file:

1. **Open the file** on your local machine or terminal
   ```bash
   cat ~/Downloads/wordpress_html/TRANSFER_READY.html
   ```

2. **Copy the entire HTML content** (including `<style>` tags and all)

3. **Log into WordPress**
   - Go to: https://synexs.net/wp-admin
   - Username: synexs

4. **Create New Post/Page**
   - Click **Posts → Add New** (or **Pages → Add New**)
   - Enter the title (e.g., "Transfer Ready")

5. **Switch to Code Editor**
   - Click the three dots menu (⋮) in the top right
   - Select **Code editor**

6. **Paste the HTML**
   - Delete any default content
   - Paste the full HTML from the file
   - The styling will be preserved

7. **Publish or Save as Draft**

#### Option B: Download Files to Local Machine

If you're working from a remote server, download the files first:

```bash
# From your local machine, run:
scp -r root@your-target.com:~/Downloads/wordpress_html/ ~/Desktop/synexs_docs/
```

Then open each HTML file in a text editor and copy/paste into WordPress.

#### Option C: Use WordPress File Upload (if available)

Some WordPress setups allow direct HTML file uploads:

1. Go to **Media → Add New**
2. Upload the `.html` files
3. However, this may not render properly - Option A is better

### Recommended Upload Order

1. **SYNEXS_MASTER_DOCUMENTATION.html** - Start with the main overview
2. **BINARY_PROTOCOL_COMPLETE.html** - Core protocol documentation
3. **BINARY_PROTOCOL_DEPLOYMENT.html** - Deployment guide
4. **GPU_SETUP_README.html** - Technical setup
5. **TRANSFER_READY.html** - Transfer checklist

### Tips

- **Use Pages instead of Posts** if you want static documentation (no dates shown)
- **Create a Documentation category** to organize all files
- **Save as Drafts first** to review how they look before publishing
- **The HTML includes embedded CSS** so styling will be preserved
- **Consider creating a Documentation menu** linking all pages together

### Troubleshooting

**If styling doesn't appear:**
- Make sure you're in **Code Editor** mode, not Visual Editor
- Ensure the `<style>` tags at the top of each file are included
- Your WordPress theme might override some styles

**If you prefer WordPress blocks:**
- Use a **Custom HTML block** for each file's content
- This gives you more flexibility to add other WordPress elements

### Next Steps

After uploading:
1. Review each page/post to ensure styling is correct
2. Add tags/categories for better organization
3. Create internal links between related documents
4. Add to your site's main navigation menu
5. Set up SEO titles and descriptions

---

**Files are located at:** `/root/Downloads/wordpress_html/`

**Quick command to view a file:**
```bash
cat ~/Downloads/wordpress_html/SYNEXS_MASTER_DOCUMENTATION.html
```

**Quick command to copy all files to clipboard (if you have xclip):**
```bash
cat ~/Downloads/wordpress_html/TRANSFER_READY.html | xclip -selection clipboard
```
