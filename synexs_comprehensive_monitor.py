#!/usr/bin/env python3
"""
Synexs Comprehensive System Monitor & Verification
Monitors all processes, datasets, GPU training, and provides improvement suggestions
"""

import os
import sys
import json
import time
import psutil
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Configuration
WORK_DIR = "/root/synexs" if os.path.exists("/root/synexs") else "/app"
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8204790720:AAEFxHurgJGIigQh0MtOlUbxX46PU8A2rA8')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '1749138955')
STATE_FILE = os.path.join(WORK_DIR, '.monitor_state.json')
LOG_FILE = os.path.join(WORK_DIR, 'comprehensive_monitor.log')

# Critical processes to monitor
CRITICAL_PROCESSES = {
    'listener.py': {'max_instances': 2, 'required': True},
    'honeypot_server.py': {'max_instances': 1, 'required': True},
    'ai_swarm_fixed.py': {'max_instances': 1, 'required': True},
    'synexs_core_orchestrator.py': {'max_instances': 1, 'required': False},
    'synexs_core_loop2.0.py': {'max_instances': 1, 'required': False},
    'propagate_v3.py': {'max_instances': 1, 'required': False},
}

# Thresholds (optimized to be more conservative)
MAX_CPU_PERCENT = 70
MAX_MEMORY_PERCENT = 85
MAX_DISK_PERCENT = 90
MAX_LOG_SIZE_MB = 50
DATASET_GROWTH_THRESHOLD_MB = 10

def log(message: str, level: str = "INFO"):
    """Log message to file and stdout"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] [{level}] {message}"
    print(log_msg, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_msg + "\n")
    except Exception as e:
        print(f"Error writing to log: {e}")

def send_telegram(message: str) -> bool:
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
        log(f"Telegram error: {e}", "ERROR")
        return False

def load_state() -> Dict:
    """Load previous monitoring state"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        log(f"Error loading state: {e}", "ERROR")
    return {
        'last_check': None,
        'dataset_sizes': {},
        'training_runs': [],
        'last_training': None,
        'alerts_sent': 0
    }

def save_state(state: Dict):
    """Save monitoring state"""
    try:
        state['last_check'] = datetime.now().isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"Error saving state: {e}", "ERROR")

def check_processes() -> Dict:
    """Check all critical processes"""
    results = {
        'status': '✅ OK',
        'running': [],
        'missing': [],
        'excess': [],
        'high_cpu': [],
        'alerts': []
    }

    process_counts = {name: [] for name in CRITICAL_PROCESSES.keys()}

    # Count running processes (optimized: get CPU in one pass)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            proc_name_lower = proc.info.get('name', '').lower()

            # Skip shell wrappers (bash, sh running our scripts)
            if proc_name_lower in ['bash', 'sh'] and any(p in cmdline for p in ['source', 'eval', 'claude']):
                continue

            for proc_name in CRITICAL_PROCESSES.keys():
                # Match both "listener.py" and "-m listener" patterns
                base_name = proc_name.replace('.py', '')
                if (proc_name in cmdline or f"-m {base_name}" in cmdline) and 'python' in cmdline:
                    # Use cached cpu_percent from process_iter (no blocking)
                    cpu = proc.info.get('cpu_percent', 0.0) or 0.0
                    process_counts[proc_name].append({
                        'pid': proc.info['pid'],
                        'cpu': cpu
                    })
                    if cpu > 50:
                        results['high_cpu'].append(f"{proc_name} (PID {proc.info['pid']}): {cpu:.1f}%")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Analyze results
    for proc_name, config in CRITICAL_PROCESSES.items():
        count = len(process_counts[proc_name])
        max_allowed = config['max_instances']
        required = config['required']

        if count == 0 and required:
            results['missing'].append(proc_name)
            results['alerts'].append(f"⛔ <b>{proc_name}</b> is not running (REQUIRED)")
        elif count > max_allowed:
            results['excess'].append(f"{proc_name} ({count} instances, max {max_allowed})")
            results['alerts'].append(f"⚠️ <b>{proc_name}</b>: {count} instances (max: {max_allowed})")
        elif count > 0:
            results['running'].append(proc_name)

    if results['high_cpu']:
        results['alerts'].append(f"🔥 <b>High CPU Usage</b>:\n" + "\n".join(results['high_cpu'][:3]))

    if results['missing'] or results['excess']:
        results['status'] = '⚠️ ISSUES'

    return results

def check_docker_containers() -> Dict:
    """Check Docker container status"""
    results = {
        'status': '✅ OK',
        'containers': [],
        'alerts': []
    }

    try:
        import subprocess
        output = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}\\t{{.Status}}'],
                                        text=True, stderr=subprocess.DEVNULL)

        for line in output.strip().split('\n'):
            if not line or 'synexs' not in line.lower():
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                name, status = parts[0], parts[1]
                results['containers'].append({'name': name, 'status': status})
    except:
        pass  # Docker not available or not running

    return results

def check_system_resources() -> Dict:
    """Check CPU, memory, disk usage"""
    results = {
        'status': '✅ OK',
        'cpu': 0,
        'memory': 0,
        'disk': 0,
        'alerts': []
    }

    try:
        # Take 2 readings 1 second apart for more accurate CPU measurement
        psutil.cpu_percent(interval=None)  # Discard first reading
        time.sleep(1)
        results['cpu'] = psutil.cpu_percent(interval=None)

        results['memory'] = psutil.virtual_memory().percent
        results['disk'] = psutil.disk_usage('/').percent

        # Only alert if CPU is sustained high (above threshold)
        # Don't alert for temporary spikes from monitoring itself
        if results['cpu'] > MAX_CPU_PERCENT:
            # Double-check with another reading
            time.sleep(1)
            cpu_check2 = psutil.cpu_percent(interval=None)
            if cpu_check2 > MAX_CPU_PERCENT:
                results['status'] = '⚠️ HIGH'
                results['alerts'].append(f"CPU: {results['cpu']:.1f}% sustained (threshold: {MAX_CPU_PERCENT}%)")

        if results['memory'] > MAX_MEMORY_PERCENT:
            results['status'] = '⚠️ HIGH'
            results['alerts'].append(f"Memory: {results['memory']:.1f}% (threshold: {MAX_MEMORY_PERCENT}%)")

        if results['disk'] > MAX_DISK_PERCENT:
            results['status'] = '⚠️ HIGH'
            results['alerts'].append(f"Disk: {results['disk']:.1f}% (threshold: {MAX_DISK_PERCENT}%)")

    except Exception as e:
        log(f"Error checking resources: {e}", "ERROR")
        results['status'] = '❌ ERROR'

    return results

def check_datasets(state: Dict) -> Dict:
    """Check dataset health and growth"""
    results = {
        'status': '✅ OK',
        'datasets': {},
        'total_size_mb': 0,
        'growth_mb': 0,
        'alerts': [],
        'recommendations': []
    }

    dataset_dirs = [
        'datasets/honeypot',
        'datasets/agents',
        'datasets/models',
        'training_logs',
    ]

    current_sizes = {}

    for dataset_dir in dataset_dirs:
        full_path = os.path.join(WORK_DIR, dataset_dir)
        if not os.path.exists(full_path):
            continue

        total_size = 0
        file_count = 0

        for root, dirs, files in os.walk(full_path):
            for f in files:
                try:
                    fp = os.path.join(root, f)
                    size = os.path.getsize(fp)
                    total_size += size
                    file_count += 1
                except:
                    continue

        size_mb = total_size / (1024 * 1024)
        current_sizes[dataset_dir] = size_mb
        results['datasets'][dataset_dir] = {
            'size_mb': round(size_mb, 2),
            'file_count': file_count
        }
        results['total_size_mb'] += size_mb

    # Check growth compared to last check
    if 'dataset_sizes' in state and state['dataset_sizes']:
        for dataset_dir, current_size in current_sizes.items():
            old_size = state['dataset_sizes'].get(dataset_dir, 0)
            growth = current_size - old_size
            results['growth_mb'] += growth

            if growth < DATASET_GROWTH_THRESHOLD_MB and current_size > 0:
                results['recommendations'].append(
                    f"📊 {dataset_dir}: Low growth ({growth:.1f}MB). Consider running data collection."
                )

    # Update state
    state['dataset_sizes'] = current_sizes

    # Check for honeypot attacks
    attacks_file = os.path.join(WORK_DIR, 'datasets/honeypot/attacks.json')
    if os.path.exists(attacks_file):
        try:
            with open(attacks_file, 'r') as f:
                attacks = [line for line in f if line.strip()]
                results['honeypot_attacks'] = len(attacks)

                if len(attacks) > 10000:
                    results['recommendations'].append(
                        f"🎯 {len(attacks)} honeypot attacks logged. Good dataset for training!"
                    )
        except Exception as e:
            log(f"Error checking honeypot attacks: {e}", "ERROR")

    return results

def check_training_status(state: Dict) -> Dict:
    """Check GPU training status and readiness"""
    results = {
        'status': '✅ READY',
        'gpu_available': False,
        'last_training': None,
        'days_since_training': None,
        'training_data_ready': False,
        'alerts': [],
        'recommendations': []
    }

    # Check GPU availability
    try:
        import subprocess
        nvidia_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                               text=True, stderr=subprocess.DEVNULL)
        if nvidia_output.strip():
            results['gpu_available'] = True
            results['gpu_info'] = nvidia_output.strip().split(',')[0]
    except Exception as e:
        log(f"Error checking GPU: {e}", "ERROR")
        results['recommendations'].append(
            "💻 GPU not detected. Training will use CPU (slower)."
        )

    # Check training data availability
    training_logs = os.path.join(WORK_DIR, 'training_logs')
    if os.path.exists(training_logs):
        batch_files = list(Path(training_logs).glob('**/batch_*.pt'))
        if len(batch_files) >= 10:
            results['training_data_ready'] = True
            results['batch_count'] = len(batch_files)
            results['recommendations'].append(
                f"🎓 {len(batch_files)} training batches ready! Run: python3 synexs_gpu_trainer.py"
            )

    # Check last training run
    if state.get('last_training'):
        last_training = datetime.fromisoformat(state['last_training'])
        days_since = (datetime.now() - last_training).days
        results['last_training'] = state['last_training']
        results['days_since_training'] = days_since

        if days_since > 7:
            results['status'] = '⚠️ OVERDUE'
            results['recommendations'].append(
                f"📅 Last training was {days_since} days ago. Consider retraining the model."
            )
    else:
        results['recommendations'].append(
            "🆕 No training runs recorded. Start with: python3 synexs_phase1_runner.py --missions 1000"
        )

    return results

def check_log_health() -> Dict:
    """Check log file sizes and health"""
    results = {
        'status': '✅ OK',
        'oversized': [],
        'total_size_mb': 0,
        'alerts': []
    }

    for log_file in Path(WORK_DIR).glob("*.log*"):
        try:
            size_mb = log_file.stat().st_size / (1024 * 1024)
            results['total_size_mb'] += size_mb

            if size_mb > MAX_LOG_SIZE_MB:
                results['oversized'].append(f"{log_file.name}: {size_mb:.1f}MB")
        except Exception as e:
            log(f"Error checking log file {log_file}: {e}", "ERROR")

    if results['oversized']:
        results['status'] = '⚠️ LARGE'
        results['alerts'].append(
            f"📁 Large log files detected:\n" + "\n".join(results['oversized'][:3])
        )

    return results

def generate_improvement_suggestions(all_results: Dict) -> List[str]:
    """Generate actionable improvement suggestions"""
    suggestions = []

    # Dataset suggestions
    if all_results['datasets']['total_size_mb'] < 100:
        suggestions.append(
            "📈 Dataset Size: Total datasets < 100MB. Run honeypot and agent collection to gather more data."
        )

    # Process suggestions
    if all_results['processes']['missing']:
        suggestions.append(
            f"🔄 Restart missing processes: {', '.join(all_results['processes']['missing'])}"
        )

    # Resource suggestions
    if all_results['resources']['cpu'] > MAX_CPU_PERCENT:
        suggestions.append(
            f"⚡ High CPU usage ({all_results['resources']['cpu']:.1f}%). Consider optimizing or scaling."
        )

    if all_results['resources']['memory'] > MAX_MEMORY_PERCENT:
        suggestions.append(
            f"💾 High memory usage ({all_results['resources']['memory']:.1f}%). Check for memory leaks."
        )

    # Training suggestions from dataset check
    if 'recommendations' in all_results.get('datasets', {}):
        suggestions.extend(all_results['datasets']['recommendations'])

    # Training suggestions from training check
    if 'recommendations' in all_results.get('training', {}):
        suggestions.extend(all_results['training']['recommendations'])

    return suggestions

def main():
    """Main monitoring loop"""
    log("=" * 60)
    log("Synexs Comprehensive Monitor Starting")
    log("=" * 60)

    state = load_state()
    all_results = {}

    # Run all checks
    log("Checking processes...")
    all_results['processes'] = check_processes()

    log("Checking system resources...")
    all_results['resources'] = check_system_resources()

    log("Checking datasets...")
    all_results['datasets'] = check_datasets(state)

    log("Checking training status...")
    all_results['training'] = check_training_status(state)

    log("Checking log health...")
    all_results['logs'] = check_log_health()

    log("Checking Docker containers...")
    all_results['docker'] = check_docker_containers()

    # Generate report
    alerts = []
    for category, results in all_results.items():
        if results.get('alerts'):
            alerts.extend(results['alerts'])

    suggestions = generate_improvement_suggestions(all_results)

    # Build summary report
    report_lines = [
        "🔍 <b>Synexs System Monitor Report</b>",
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"🖥️ <b>Processes:</b> {all_results['processes']['status']}",
        f"  Running: {', '.join(all_results['processes']['running'][:5]) or 'None'}",
    ]

    if all_results['processes']['missing']:
        report_lines.append(f"  ⛔ Missing: {', '.join(all_results['processes']['missing'])}")

    if all_results['processes']['excess']:
        report_lines.append(f"  ⚠️ Excess: {', '.join(all_results['processes']['excess'])}")

    report_lines.extend([
        "",
        f"💻 <b>Resources:</b> {all_results['resources']['status']}",
        f"  CPU: {all_results['resources']['cpu']:.1f}% | Memory: {all_results['resources']['memory']:.1f}% | Disk: {all_results['resources']['disk']:.1f}%",
        "",
        f"📊 <b>Datasets:</b> {all_results['datasets']['total_size_mb']:.1f}MB total",
    ])

    for dataset_name, dataset_info in list(all_results['datasets']['datasets'].items())[:3]:
        report_lines.append(f"  {dataset_name}: {dataset_info['size_mb']:.1f}MB ({dataset_info['file_count']} files)")

    if all_results['training'].get('gpu_available'):
        report_lines.append(f"\n🎮 <b>GPU:</b> {all_results['training'].get('gpu_info', 'Available')}")

    if all_results['training'].get('training_data_ready'):
        report_lines.append(f"🎓 <b>Training:</b> {all_results['training'].get('batch_count', 0)} batches ready")

    # Add alerts
    if alerts:
        report_lines.append("\n⚠️ <b>ALERTS:</b>")
        for alert in alerts[:5]:
            report_lines.append(f"  {alert}")

    # Add suggestions
    if suggestions:
        report_lines.append("\n💡 <b>SUGGESTIONS:</b>")
        for suggestion in suggestions[:5]:
            report_lines.append(f"  {suggestion}")

    report = "\n".join(report_lines)

    # Log summary
    log("\n" + report.replace('<b>', '').replace('</b>', ''))

    # Send alerts if there are critical issues
    if alerts:
        send_telegram(report)
        state['alerts_sent'] = state.get('alerts_sent', 0) + 1

    # Save state
    save_state(state)

    log("=" * 60)
    log("Monitor check complete")
    log("=" * 60)

if __name__ == "__main__":
    main()