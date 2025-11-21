#!/usr/bin/env python3
"""
SYNEXS Pipeline Manager
Manages all pipeline processes, health checks, and auto-restart functionality
"""

import json
import subprocess
import sys
import time
import os
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

CONFIG_FILE = "/root/synexs/pipeline_config.json"
LOG_DIR = "/root/synexs/logs"
PID_DIR = "/root/synexs/pids"

# Ensure directories exist
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(PID_DIR).mkdir(parents=True, exist_ok=True)


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def load_config() -> Dict:
    """Load pipeline configuration from JSON file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Colors.RED}✗ Configuration file not found: {CONFIG_FILE}{Colors.RESET}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}✗ Invalid JSON in config file: {e}{Colors.RESET}")
        sys.exit(1)


def save_config(config: Dict):
    """Save pipeline configuration to JSON file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"{Colors.GREEN}✓ Configuration saved{Colors.RESET}")


def is_process_running(name: str) -> Optional[int]:
    """Check if a process is running by name, return PID or None"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            # Return first PID found
            return int(result.stdout.strip().split('\n')[0])
        return None
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Error checking process {name}: {e}{Colors.RESET}")
        return None


def check_http_health(url: str, expect_key: str = None, expect_value: str = None) -> bool:
    """Check HTTP health endpoint"""
    try:
        import requests
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return False

        if expect_key and expect_value:
            data = response.json()
            return data.get(expect_key) == expect_value
        return True
    except Exception as e:
        return False


def check_redis_health() -> bool:
    """Check if Redis is running and responding"""
    try:
        result = subprocess.run(
            ['redis-cli', 'ping'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and 'PONG' in result.stdout
    except Exception:
        return False


def start_process(process: Dict) -> bool:
    """Start a process based on configuration"""
    name = process['name']
    script = process['script']
    python = process['python']
    working_dir = process['working_dir']
    args = process.get('args', [])

    print(f"{Colors.BLUE}▶ Starting {name}...{Colors.RESET}")

    try:
        # Build command
        if args and args[0] == '-m':
            # Module execution (like python3 -m listener)
            cmd = [python] + args + [script]
        elif args:
            # Script with args
            cmd = [python] + args + [script]
        else:
            # Simple script execution
            cmd = [python, script]

        # Log file for this process
        log_file = f"{LOG_DIR}/{name}.log"

        # Start process in background
        with open(log_file, 'a') as log:
            log.write(f"\n{'='*60}\n")
            log.write(f"Started at: {datetime.now().isoformat()}\n")
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write(f"{'='*60}\n\n")

            proc = subprocess.Popen(
                cmd,
                cwd=working_dir,
                stdout=log,
                stderr=log,
                start_new_session=True
            )

        # Give it a moment to start
        time.sleep(2)

        # Check if it's running
        if proc.poll() is None:
            # Save PID
            pid_file = f"{PID_DIR}/{name}.pid"
            with open(pid_file, 'w') as f:
                f.write(str(proc.pid))

            print(f"{Colors.GREEN}✓ {name} started (PID: {proc.pid}){Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}✗ {name} failed to start{Colors.RESET}")
            print(f"  Check logs: {log_file}")
            return False

    except Exception as e:
        print(f"{Colors.RED}✗ Error starting {name}: {e}{Colors.RESET}")
        return False


def stop_process(name: str, pid: int) -> bool:
    """Stop a running process"""
    try:
        print(f"{Colors.YELLOW}◼ Stopping {name} (PID: {pid})...{Colors.RESET}")
        os.kill(pid, signal.SIGTERM)

        # Wait up to 10 seconds for graceful shutdown
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except OSError:
                print(f"{Colors.GREEN}✓ {name} stopped{Colors.RESET}")
                return True

        # Force kill if still running
        print(f"{Colors.YELLOW}⚠ Force killing {name}...{Colors.RESET}")
        os.kill(pid, signal.SIGKILL)
        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Error stopping {name}: {e}{Colors.RESET}")
        return False


def check_status(verbose: bool = False) -> Dict:
    """Check status of all processes and services"""
    config = load_config()
    status = {
        'processes': [],
        'services': [],
        'all_healthy': True
    }

    # Check processes
    for proc in config.get('processes', []):
        if not proc.get('enabled', True):
            continue

        name = proc['name']
        pid = is_process_running(name)

        proc_status = {
            'name': name,
            'running': pid is not None,
            'pid': pid,
            'health': None
        }

        # Check health if running and health check configured
        if pid and proc.get('health_check'):
            hc = proc['health_check']
            if hc['type'] == 'http':
                proc_status['health'] = check_http_health(
                    hc['url'],
                    hc.get('expect_key'),
                    hc.get('expect_value')
                )

        if not proc_status['running'] or (proc_status['health'] is not None and not proc_status['health']):
            status['all_healthy'] = False

        status['processes'].append(proc_status)

    # Check services
    for svc in config.get('services', []):
        name = svc['name']
        svc_status = {
            'name': name,
            'type': svc['type'],
            'health': False
        }

        if name == 'redis':
            svc_status['health'] = check_redis_health()

        if not svc_status['health']:
            status['all_healthy'] = False

        status['services'].append(svc_status)

    return status


def print_status(status: Dict):
    """Pretty print status report"""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}  SYNEXS PIPELINE STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    # Print processes
    print(f"{Colors.BOLD}PROCESSES:{Colors.RESET}")
    for proc in status['processes']:
        status_icon = f"{Colors.GREEN}✓{Colors.RESET}" if proc['running'] else f"{Colors.RED}✗{Colors.RESET}"
        name = proc['name']

        if proc['running']:
            pid_info = f"PID: {proc['pid']}"
            if proc['health'] is not None:
                health_icon = f"{Colors.GREEN}✓{Colors.RESET}" if proc['health'] else f"{Colors.RED}✗{Colors.RESET}"
                health_status = f"Health: {health_icon}"
            else:
                health_status = ""
            print(f"  {status_icon} {name:<30} {pid_info:<15} {health_status}")
        else:
            print(f"  {status_icon} {name:<30} {Colors.RED}NOT RUNNING{Colors.RESET}")

    # Print services
    if status['services']:
        print(f"\n{Colors.BOLD}SERVICES:{Colors.RESET}")
        for svc in status['services']:
            status_icon = f"{Colors.GREEN}✓{Colors.RESET}" if svc['health'] else f"{Colors.RED}✗{Colors.RESET}"
            print(f"  {status_icon} {svc['name']:<30} {svc['type']}")

    # Overall status
    print(f"\n{Colors.BOLD}PIPELINE:{Colors.RESET}", end=" ")
    if status['all_healthy']:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ FULLY OPERATIONAL{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ ISSUES DETECTED{Colors.RESET}")

    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")


def start_all():
    """Start all enabled processes"""
    config = load_config()
    print(f"\n{Colors.BOLD}Starting all pipeline processes...{Colors.RESET}\n")

    for proc in config.get('processes', []):
        if not proc.get('enabled', True):
            print(f"{Colors.YELLOW}⊘ {proc['name']} (disabled){Colors.RESET}")
            continue

        pid = is_process_running(proc['name'])
        if pid:
            print(f"{Colors.GREEN}✓ {proc['name']} already running (PID: {pid}){Colors.RESET}")
        else:
            start_process(proc)

        time.sleep(1)

    print()
    status = check_status()
    print_status(status)


def stop_all():
    """Stop all running processes"""
    config = load_config()
    print(f"\n{Colors.BOLD}Stopping all pipeline processes...{Colors.RESET}\n")

    for proc in config.get('processes', []):
        pid = is_process_running(proc['name'])
        if pid:
            stop_process(proc['name'], pid)
        else:
            print(f"{Colors.YELLOW}⊘ {proc['name']} not running{Colors.RESET}")


def restart_all():
    """Restart all processes"""
    stop_all()
    time.sleep(2)
    start_all()


def add_process():
    """Interactive process addition"""
    config = load_config()

    print(f"\n{Colors.BOLD}Add New Process{Colors.RESET}\n")

    name = input("Process name: ").strip()
    script = input("Script path: ").strip()
    python = input("Python path [/usr/bin/python3]: ").strip() or "/usr/bin/python3"
    working_dir = input("Working directory [/root/synexs]: ").strip() or "/root/synexs"
    args = input("Arguments (space-separated, optional): ").strip()

    new_process = {
        "name": name,
        "script": script,
        "python": python,
        "working_dir": working_dir,
        "args": args.split() if args else [],
        "enabled": True,
        "health_check": None
    }

    # Ask about health check
    add_health = input("Add HTTP health check? (y/n): ").strip().lower()
    if add_health == 'y':
        url = input("Health check URL: ").strip()
        expect_key = input("Expected JSON key (optional): ").strip() or None
        expect_value = input("Expected value (optional): ").strip() or None

        new_process["health_check"] = {
            "type": "http",
            "url": url,
            "expect_key": expect_key,
            "expect_value": expect_value
        }

    config['processes'].append(new_process)
    save_config(config)
    print(f"{Colors.GREEN}✓ Process '{name}' added to configuration{Colors.RESET}")


def remove_process():
    """Interactive process removal"""
    config = load_config()

    print(f"\n{Colors.BOLD}Remove Process{Colors.RESET}\n")

    for i, proc in enumerate(config.get('processes', []), 1):
        print(f"{i}. {proc['name']}")

    choice = input("\nEnter number to remove (or 'cancel'): ").strip()

    if choice.lower() == 'cancel':
        return

    try:
        idx = int(choice) - 1
        removed = config['processes'].pop(idx)
        save_config(config)
        print(f"{Colors.GREEN}✓ Process '{removed['name']}' removed{Colors.RESET}")
    except (ValueError, IndexError):
        print(f"{Colors.RED}✗ Invalid selection{Colors.RESET}")


def list_processes():
    """List all configured processes"""
    config = load_config()

    print(f"\n{Colors.BOLD}Configured Processes:{Colors.RESET}\n")

    for proc in config.get('processes', []):
        enabled = "✓" if proc.get('enabled', True) else "✗"
        print(f"  [{enabled}] {proc['name']}")
        print(f"      Script: {proc['script']}")
        print(f"      Python: {proc['python']}")
        print(f"      Working Dir: {proc['working_dir']}")
        if proc.get('health_check'):
            print(f"      Health Check: {proc['health_check']['url']}")
        print()


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(f"\n{Colors.BOLD}SYNEXS Pipeline Manager{Colors.RESET}\n")
        print("Usage: pipeline_manager.py <command>\n")
        print("Commands:")
        print("  status       - Show status of all processes")
        print("  start        - Start all enabled processes")
        print("  stop         - Stop all processes")
        print("  restart      - Restart all processes")
        print("  add          - Add a new process")
        print("  remove       - Remove a process")
        print("  list         - List all configured processes")
        print("  monitor      - Continuous monitoring (check every 30s)")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'status':
        status = check_status()
        print_status(status)
        sys.exit(0 if status['all_healthy'] else 1)

    elif command == 'start':
        start_all()

    elif command == 'stop':
        stop_all()

    elif command == 'restart':
        restart_all()

    elif command == 'add':
        add_process()

    elif command == 'remove':
        remove_process()

    elif command == 'list':
        list_processes()

    elif command == 'monitor':
        print(f"{Colors.BOLD}Monitoring pipeline (Ctrl+C to stop)...{Colors.RESET}\n")
        try:
            while True:
                status = check_status()
                print_status(status)

                # Auto-restart failed processes
                if not status['all_healthy']:
                    print(f"{Colors.YELLOW}⚠ Detected issues, attempting auto-restart...{Colors.RESET}\n")
                    config = load_config()
                    for proc_status in status['processes']:
                        if not proc_status['running']:
                            # Find config for this process
                            proc_config = next((p for p in config['processes'] if p['name'] == proc_status['name']), None)
                            if proc_config and proc_config.get('enabled', True):
                                start_process(proc_config)

                time.sleep(30)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitoring stopped{Colors.RESET}")

    else:
        print(f"{Colors.RED}Unknown command: {command}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
