#!/bin/bash
# Display WordPress HTML content for manual upload

cd ~/Downloads/wordpress_html

files=(
    "TRANSFER_READY.html"
    "BINARY_PROTOCOL_COMPLETE.html"
    "BINARY_PROTOCOL_DEPLOYMENT.html"
    "GPU_SETUP_README.html"
    "SYNEXS_MASTER_DOCUMENTATION.html"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "📄 FILE: $file"
        echo "📝 TITLE: $(echo "$file" | sed 's/.html$//' | sed 's/_/ /g')"
        echo "📊 SIZE: $(wc -c < "$file" | numfmt --to=iec-i --suffix=B 2>/dev/null || echo "$(wc -c < "$file") bytes")"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo ">>> COPY THE CONTENT BELOW <<<"
        echo ""
        cat "$file"
        echo ""
        echo ""
        echo ">>> END OF $file <<<"
        echo ""
        read -p "Press ENTER to see next file..."
    fi
done

echo ""
echo "✅ All files displayed!"
