#!/usr/bin/env python3
"""
Report Server - Defensive Research Tool
Purpose: Receive and log agent death reports for AI training
Authorization: Owner-controlled systems only
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import threading
import time

# Configuration
PORT = 8080
LOG_DIR = "/root/synexs/datasets"
KILL_LOG = os.path.join(LOG_DIR, "real_world_kills.json")
STATS_FILE = os.path.join(LOG_DIR, "kill_stats.json")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Flask app
app = Flask(__name__)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('report_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReportServer:
    def __init__(self):
        self.reports = []
        self.stats = {
            "total_reports": 0,
            "blocked_count": 0,
            "survived_avg": 0.0,
            "av_detections": {},
            "death_reasons": {},
            "os_breakdown": {}
        }
        self.load_existing_reports()

    def load_existing_reports(self):
        """Load existing reports from disk"""
        try:
            if os.path.exists(KILL_LOG):
                with open(KILL_LOG, 'r') as f:
                    self.reports = json.load(f)
                logger.info(f"Loaded {len(self.reports)} existing reports")
        except Exception as e:
            logger.error(f"Error loading reports: {e}")
            self.reports = []

    def save_reports(self):
        """Save reports to disk"""
        try:
            with open(KILL_LOG, 'w') as f:
                json.dump(self.reports, f, indent=2)
            logger.info(f"Saved {len(self.reports)} reports to {KILL_LOG}")
        except Exception as e:
            logger.error(f"Error saving reports: {e}")

    def update_stats(self):
        """Calculate statistics from reports"""
        if not self.reports:
            return

        self.stats = {
            "total_reports": len(self.reports),
            "blocked_count": sum(1 for r in self.reports if r.get("blocked")),
            "survived_avg": sum(r.get("survived_seconds", 0) for r in self.reports) / len(self.reports),
            "av_detections": {},
            "death_reasons": {},
            "os_breakdown": {}
        }

        # Count AV detections
        for report in self.reports:
            av_info = report.get("av_status", {})
            for av in av_info.get("detected", []):
                self.stats["av_detections"][av] = self.stats["av_detections"].get(av, 0) + 1

            # Count death reasons
            reason = report.get("death_reason", "unknown")
            self.stats["death_reasons"][reason] = self.stats["death_reasons"].get(reason, 0) + 1

            # Count OS
            os_name = report.get("os", {}).get("system", "unknown")
            self.stats["os_breakdown"][os_name] = self.stats["os_breakdown"].get(os_name, 0) + 1

        # Save stats
        try:
            with open(STATS_FILE, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")

    def add_report(self, report_data):
        """Add a new report"""
        # Add server timestamp
        report_data["server_received_at"] = datetime.utcnow().isoformat()

        # Add to reports list
        self.reports.append(report_data)

        # Save to disk
        self.save_reports()

        # Update stats
        self.update_stats()

        logger.info(f"✓ Report received from {report_data.get('agent_id')}")
        logger.info(f"  OS: {report_data.get('os', {}).get('system')}")
        logger.info(f"  AV: {report_data.get('av_status', {}).get('detected')}")
        logger.info(f"  Survived: {report_data.get('survived_seconds', 0):.2f}s")
        logger.info(f"  Blocked: {report_data.get('blocked')}")
        logger.info(f"  Death: {report_data.get('death_reason')}")

        return True

server = ReportServer()

@app.route('/report', methods=['POST'])
def receive_report():
    """Endpoint to receive agent reports"""
    try:
        report_data = request.get_json()

        if not report_data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["agent_id", "timestamp", "os"]
        for field in required_fields:
            if field not in report_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Add report
        server.add_report(report_data)

        return jsonify({
            "status": "success",
            "message": "Report received",
            "agent_id": report_data.get("agent_id"),
            "total_reports": len(server.reports)
        }), 200

    except Exception as e:
        logger.error(f"Error processing report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get current statistics"""
    return jsonify(server.stats), 200

@app.route('/reports', methods=['GET'])
def get_reports():
    """Get all reports"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify(server.reports[-limit:]), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "total_reports": len(server.reports),
        "uptime": time.time()
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Index page"""
    return f"""
    <html>
    <head><title>Synexs Report Server</title></head>
    <body>
        <h1>Synexs Defensive Research - Report Server</h1>
        <p>Total Reports: {len(server.reports)}</p>
        <p>Blocked: {server.stats.get('blocked_count', 0)}</p>
        <p>Avg Survival: {server.stats.get('survived_avg', 0):.2f}s</p>
        <h2>Endpoints:</h2>
        <ul>
            <li>POST /report - Submit agent report</li>
            <li>GET /stats - View statistics</li>
            <li>GET /reports?limit=N - View reports</li>
            <li>GET /health - Health check</li>
        </ul>
        <h2>AV Detections:</h2>
        <ul>
        {''.join(f'<li>{av}: {count}</li>' for av, count in server.stats.get('av_detections', {}).items())}
        </ul>
        <h2>Death Reasons:</h2>
        <ul>
        {''.join(f'<li>{reason}: {count}</li>' for reason, count in server.stats.get('death_reasons', {}).items())}
        </ul>
    </body>
    </html>
    """, 200

def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("SYNEXS REPORT SERVER - DEFENSIVE RESEARCH")
    logger.info(f"Port: {PORT}")
    logger.info(f"Log file: {KILL_LOG}")
    logger.info(f"Stats file: {STATS_FILE}")
    logger.info("="*60)

    # Run Flask server
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()