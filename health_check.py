#!/usr/bin/env python3
"""
Health Check Monitor - Synexs System
Monitors system resources and alerts on anomalies
"""

import os
import sys
import time
import psutil
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8204790720:AAEFxHurgJGIigQh0MtOlUbxX46PU8A2rA8')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '1749138955')
WORK_DIR = "/root/synexs" if os.path.exists("/root/synexs") else "/app"
LOG_FILE = os.path.join(WORK_DIR, "health_check.log")

# Thresholds
MAX_LISTENER_PROCESSES = 5
MAX_CPU_PERCENT = 80
MAX_MEMORY_PERCENT = 85
MAX_DISK_PERCENT = 90
MAX_LOG_SIZE_MB = 100

def send_telegram_alert(message: str) -> bool:
    """Send alert via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        return response.json().get("ok", False)
    except Exception as e:
        log(f"Telegram error: {e}")
        return False

def log(message: str):
    """Log to file and stdout"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_msg + "\n")
    except Exception as e:
        print(f"Error writing to log file: {e}", flush=True)

def check_listener_processes() -> Dict[str, any]:
    """Check for listener process leaks"""
    try:
        listener_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'listener.py' in cmdline and 'python' in cmdline:
                    listener_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        status = "✅ OK" if listener_count <= MAX_LISTENER_PROCESSES else "⚠️ LEAK"
        return {
            "status": status,
            "count": listener_count,
            "threshold": MAX_LISTENER_PROCESSES,
            "alert": listener_count > MAX_LISTENER_PROCESSES
        }
    except Exception as e:
        log(f"Error checking listeners: {e}")
        return {"status": "❌ ERROR", "count": 0, "alert": False}

def check_system_resources() -> Dict[str, any]:
    """Check CPU, memory, disk usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        alerts = []
        if cpu_percent > MAX_CPU_PERCENT:
            alerts.append(f"CPU at {cpu_percent:.1f}%")
        if memory_percent > MAX_MEMORY_PERCENT:
            alerts.append(f"Memory at {memory_percent:.1f}%")
        if disk_percent > MAX_DISK_PERCENT:
            alerts.append(f"Disk at {disk_percent:.1f}%")

        status = "✅ OK" if not alerts else "⚠️ HIGH"
        return {
            "status": status,
            "cpu": cpu_percent,
            "memory": memory_percent,
            "disk": disk_percent,
            "alerts": alerts,
            "alert": bool(alerts)
        }
    except Exception as e:
        log(f"Error checking resources: {e}")
        return {"status": "❌ ERROR", "alerts": [str(e)], "alert": True}

def check_log_files() -> Dict[str, any]:
    """Check for oversized log files"""
    try:
        oversized = []
        for log_file in Path(WORK_DIR).glob("*.log"):
            size_mb = log_file.stat().st_size / (1024 * 1024)
            if size_mb > MAX_LOG_SIZE_MB:
                oversized.append(f"{log_file.name}: {size_mb:.1f}MB")

        status = "✅ OK" if not oversized else "⚠️ LARGE"
        return {
            "status": status,
            "oversized": oversized,
            "alert": bool(oversized)
        }
    except Exception as e:
        log(f"Error checking logs: {e}")
        return {"status": "❌ ERROR", "oversized": [str(e)], "alert": True}

def check_critical_processes() -> Dict[str, any]:
    """Check if critical processes are running"""
    critical_processes = [
        'honeypot_server.py',
        'listener.py',
        'propagate_v3.py',
        'ai_swarm_fixed.py'
    ]

    running, missing = [], []

    for proc in psutil.process_iter(['cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            for critical in critical_processes:
                if critical in cmdline and critical not in running:
                    running.append(critical)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for critical in critical_processes:
        if critical not in running:
            missing.append(critical)

    status = "✅ OK" if not missing else "⚠️ DOWN"
    return {
        "status": status,
        "running": running,
        "missing": missing,
        "alert": bool(missing)
    }

def run_health_check() -> Dict[str, any]:
    """Run all health checks and alert if issues found"""
    log("=" * 60)
    log("🏥 Health Check Started")
    log("=" * 60)

    # Run checks
    listener_check = check_listener_processes()
    resource_check = check_system_resources()
    log_check = check_log_files()
    process_check = check_critical_processes()

    # Log results
    log(f"Listeners: {listener_check['status']} ({listener_check['count']} running)")
    log(f"Resources: {resource_check['status']} (CPU: {resource_check.get('cpu', 0):.1f}%, "
        f"Mem: {resource_check.get('memory', 0):.1f}%, Disk: {resource_check.get('disk', 0):.1f}%)")
    log(f"Log Files: {log_check['status']}")
    log(f"Processes: {process_check['status']} ({len(process_check['running'])}/{len(process_check['running']) + len(process_check['missing'])} running)")

    # Send alerts if needed
    alerts = []

    if listener_check['alert']:
        alerts.append(f"🚨 <b>Listener Leak</b>\n{listener_check['count']} processes running (max: {MAX_LISTENER_PROCESSES})")

    if resource_check['alert']:
        alerts.append(f"⚠️ <b>High Resource Usage</b>\n" + "\n".join(resource_check['alerts']))

    if log_check['alert']:
        alerts.append(f"📁 <b>Large Log Files</b>\n" + "\n".join(log_check['oversized'][:3]))

    if process_check['alert']:
        alerts.append(f"⛔ <b>Processes Down</b>\n" + "\n".join(process_check['missing']))

    if alerts:
        alert_message = f"🏥 <b>Synexs Health Alert</b>\n\n" + "\n\n".join(alerts)
        log("Sending alert via Telegram...")
        if not send_telegram_alert(alert_message):
            log("Failed to send Telegram alert")

    log("Health check completed.\n")

    # Return summary
    return {
        "timestamp": datetime.now().isoformat(),
        "listener": listener_check,
        "resources": resource_check,
        "logs": log_check,
        "processes": process_check,
        "alerts_sent": len(alerts)
    }

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == '--loop':
        log("Health check monitor starting (loop mode)...")
        while True:
            try:
                run_health_check()
                time.sleep(300)  # Check every 5 minutes
            except KeyboardInterrupt:
                log("Shutting down...")
                break
            except Exception as e:
                log(f"Error in main loop: {e}")
                time.sleep(60)
    else:
        # Single run
        result = run_health_check()
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()