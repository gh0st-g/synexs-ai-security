#!/usr/bin/env python3
"""
Synexs Monitoring Verification Script
Quick check of all critical processes and resources
"""

import psutil
import subprocess
import json
from datetime import datetime


def colored(text, color):
    """Simple color output"""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"


def check_process(name, pattern=None):
    """Check if a process is running"""
    if pattern is None:
        pattern = name

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            proc_name_lower = proc.info.get('name', '').lower()

            # Skip shell wrappers
            if proc_name_lower in ['bash', 'sh'] and any(p in cmdline for p in ['source', 'eval', 'claude']):
                continue

            # Check for both script name and module pattern
            base_name = pattern.replace('.py', '')
            if (pattern in cmdline or f"-m {base_name}" in cmdline) and 'python' in cmdline:
                return {
                    'running': True,
                    'pid': proc.info['pid'],
                    'cpu': proc.cpu_percent(interval=0.1)
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {'running': False, 'pid': None, 'cpu': 0}


def check_health_endpoint(port):
    """Check if health endpoint is responding"""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def get_cpu_average():
    """Get average CPU over 2 seconds"""
    psutil.cpu_percent(interval=None)  # Discard first reading
    import time
    time.sleep(1)
    cpu1 = psutil.cpu_percent(interval=None)
    time.sleep(1)
    cpu2 = psutil.cpu_percent(interval=None)
    return (cpu1 + cpu2) / 2


def main():
    print("=" * 70)
    print(f"  SYNEXS MONITORING VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Critical processes to check
    processes = {
        'listener.py': {'port': 8765, 'required': True},
        'honeypot_server.py': {'port': 8080, 'required': True},
        'ai_swarm_fixed.py': {'required': False},
        'synexs_core_orchestrator.py': {'required': False},
    }

    print("\n📋 PROCESS STATUS:")
    all_ok = True
    for proc_name, config in processes.items():
        status = check_process(proc_name)

        if status['running']:
            icon = colored("✓", "green")
            print(f"  {icon} {proc_name:<30} PID: {status['pid']:<8} CPU: {status['cpu']:.1f}%")

            # Check health endpoint if available
            if 'port' in config:
                health_ok, health_data = check_health_endpoint(config['port'])
                if health_ok:
                    print(f"     └─ Health: {colored('OK', 'green')}")
                else:
                    print(f"     └─ Health: {colored('NO RESPONSE', 'yellow')}")
        else:
            if config['required']:
                icon = colored("✗", "red")
                all_ok = False
            else:
                icon = colored("○", "yellow")
            print(f"  {icon} {proc_name:<30} {colored('NOT RUNNING', 'red' if config['required'] else 'yellow')}")

    # System Resources
    print("\n📊 SYSTEM RESOURCES:")
    cpu_avg = get_cpu_average()
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent

    cpu_color = 'red' if cpu_avg > 85 else 'yellow' if cpu_avg > 70 else 'green'
    mem_color = 'red' if memory > 85 else 'yellow' if memory > 70 else 'green'
    disk_color = 'red' if disk > 90 else 'yellow' if disk > 80 else 'green'

    print(f"  CPU:    {colored(f'{cpu_avg:.1f}%', cpu_color)}")
    print(f"  Memory: {colored(f'{memory:.1f}%', mem_color)}")
    print(f"  Disk:   {colored(f'{disk:.1f}%', disk_color)}")

    # Redis
    print("\n🔴 REDIS:")
    try:
        result = subprocess.run(['redis-cli', 'ping'], capture_output=True, text=True, timeout=2)
        if 'PONG' in result.stdout:
            print(f"  {colored('✓', 'green')} Redis responding")
        else:
            print(f"  {colored('✗', 'red')} Redis not responding")
            all_ok = False
    except:
        print(f"  {colored('✗', 'red')} Redis not available")
        all_ok = False

    # Cron jobs
    print("\n⏰ CRON JOBS:")
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        cron_lines = result.stdout.split('\n')

        critical_jobs = {
            'propagate_v3.py': False,
            'synexs_comprehensive_monitor.py': False,
        }

        for line in cron_lines:
            for job in critical_jobs.keys():
                if job in line and not line.strip().startswith('#'):
                    critical_jobs[job] = True

        for job, found in critical_jobs.items():
            if found:
                print(f"  {colored('✓', 'green')} {job} scheduled")
            else:
                print(f"  {colored('✗', 'red')} {job} NOT scheduled")
                all_ok = False
    except Exception as e:
        print(f"  {colored('✗', 'red')} Error checking cron: {e}")

    # Final verdict
    print("\n" + "=" * 70)
    if all_ok:
        print(colored("  ✓ ALL SYSTEMS OPERATIONAL", "green"))
    else:
        print(colored("  ⚠ ISSUES DETECTED - Review above", "yellow"))
    print("=" * 70)

    # Check if comprehensive monitor exists
    print("\n🔍 MONITORING SCRIPT:")
    import os
    if os.path.exists('/root/synexs/synexs_comprehensive_monitor.py'):
        print(f"  {colored('✓', 'green')} synexs_comprehensive_monitor.py exists")

        # Test run
        print(f"\n  Testing comprehensive monitor...")
        try:
            result = subprocess.run(
                ['/root/synexs/synexs_env/bin/python3', 'synexs_comprehensive_monitor.py'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd='/root/synexs'
            )
            if result.returncode == 0:
                print(f"  {colored('✓', 'green')} Comprehensive monitor executed successfully")
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    if line.strip():
                        print(f"    {line}")
            else:
                print(f"  {colored('✗', 'red')} Comprehensive monitor failed")
                print(f"    Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  {colored('⚠', 'yellow')} Monitor taking longer than expected (still running)")
        except Exception as e:
            print(f"  {colored('✗', 'red')} Error running monitor: {e}")
    else:
        print(f"  {colored('✗', 'red')} synexs_comprehensive_monitor.py NOT FOUND")

    print()


if __name__ == "__main__":
    main()
