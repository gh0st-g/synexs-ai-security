#!/usr/bin/env python3
"""
Synexs Goals & Reminders System
Tracks weekly and monthly goals for datasets, training, and system improvements
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Configuration
WORK_DIR = "/root/synexs" if os.path.exists("/root/synexs") else "/app"
GOALS_FILE = os.path.join(WORK_DIR, '.goals_tracker.json')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8204790720:AAEFxHurgJGIigQh0MtOlUbxX46PU8A2rA8')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '1749138955')

# Default Goals Configuration
DEFAULT_GOALS = {
    'weekly': {
        'missions_generated': {
            'target': 5000,
            'description': 'Generate 5,000 training missions per week',
            'unit': 'missions',
            'priority': 'high'
        },
        'dataset_growth_mb': {
            'target': 100,
            'description': 'Grow datasets by 100MB per week',
            'unit': 'MB',
            'priority': 'medium'
        },
        'gpu_training_runs': {
            'target': 2,
            'description': 'Run GPU training at least 2 times per week',
            'unit': 'runs',
            'priority': 'high'
        },
        'honeypot_attacks': {
            'target': 1000,
            'description': 'Collect 1,000 honeypot attacks per week',
            'unit': 'attacks',
            'priority': 'medium'
        },
        'model_accuracy': {
            'target': 85,
            'description': 'Maintain model accuracy above 85%',
            'unit': '%',
            'priority': 'high'
        }
    },
    'monthly': {
        'total_missions': {
            'target': 20000,
            'description': 'Generate 20,000 total missions per month',
            'unit': 'missions',
            'priority': 'high'
        },
        'dataset_size_gb': {
            'target': 1,
            'description': 'Reach 1GB total dataset size',
            'unit': 'GB',
            'priority': 'medium'
        },
        'model_versions': {
            'target': 4,
            'description': 'Create 4 model versions per month (weekly retraining)',
            'unit': 'versions',
            'priority': 'high'
        },
        'system_uptime': {
            'target': 99,
            'description': 'Maintain 99% system uptime',
            'unit': '%',
            'priority': 'high'
        },
        'documentation_updates': {
            'target': 2,
            'description': 'Update documentation at least twice per month',
            'unit': 'updates',
            'priority': 'low'
        }
    }
}

def send_telegram(message: str) -> bool:
    """Send notification via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=5)
        return response.json().get("ok", False)
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def load_goals_state() -> Dict:
    """Load goals tracking state"""
    if os.path.exists(GOALS_FILE):
        try:
            with open(GOALS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass

    # Initialize new state
    return {
        'goals': DEFAULT_GOALS,
        'current_week': {
            'start_date': datetime.now().isoformat(),
            'progress': {goal: 0 for goal in DEFAULT_GOALS['weekly'].keys()}
        },
        'current_month': {
            'start_date': datetime.now().isoformat(),
            'progress': {goal: 0 for goal in DEFAULT_GOALS['monthly'].keys()}
        },
        'history': {
            'weekly': [],
            'monthly': []
        },
        'last_reminder': None
    }

def save_goals_state(state: Dict):
    """Save goals tracking state"""
    try:
        with open(GOALS_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving goals state: {e}")

def check_week_rollover(state: Dict) -> bool:
    """Check if we need to start a new week"""
    week_start = datetime.fromisoformat(state['current_week']['start_date'])
    days_elapsed = (datetime.now() - week_start).days

    if days_elapsed >= 7:
        # Archive current week
        state['history']['weekly'].append({
            'start_date': state['current_week']['start_date'],
            'end_date': datetime.now().isoformat(),
            'progress': state['current_week']['progress'].copy(),
            'completed': calculate_completion(state['current_week']['progress'], state['goals']['weekly'])
        })

        # Start new week
        state['current_week'] = {
            'start_date': datetime.now().isoformat(),
            'progress': {goal: 0 for goal in state['goals']['weekly'].keys()}
        }
        return True

    return False

def check_month_rollover(state: Dict) -> bool:
    """Check if we need to start a new month"""
    month_start = datetime.fromisoformat(state['current_month']['start_date'])
    days_elapsed = (datetime.now() - month_start).days

    if days_elapsed >= 30:
        # Archive current month
        state['history']['monthly'].append({
            'start_date': state['current_month']['start_date'],
            'end_date': datetime.now().isoformat(),
            'progress': state['current_month']['progress'].copy(),
            'completed': calculate_completion(state['current_month']['progress'], state['goals']['monthly'])
        })

        # Start new month
        state['current_month'] = {
            'start_date': datetime.now().isoformat(),
            'progress': {goal: 0 for goal in state['goals']['monthly'].keys()}
        }
        return True

    return False

def calculate_completion(progress: Dict, goals: Dict) -> float:
    """Calculate overall completion percentage"""
    if not goals:
        return 0.0

    total_completion = 0.0
    for goal_name, goal_config in goals.items():
        current = progress.get(goal_name, 0)
        target = goal_config['target']
        completion = min(100, (current / target * 100)) if target > 0 else 0
        total_completion += completion

    return round(total_completion / len(goals), 2)

def update_progress_from_system(state: Dict) -> Dict:
    """Update progress based on actual system metrics"""
    updates = {}

    # Check training logs for mission count
    training_logs = os.path.join(WORK_DIR, 'training_logs')
    if os.path.exists(training_logs):
        batch_files = list(Path(training_logs).glob('**/batch_*.pt'))
        missions_count = len(batch_files) * 32  # Each batch = 32 missions
        state['current_week']['progress']['missions_generated'] = missions_count
        state['current_month']['progress']['total_missions'] = missions_count
        updates['missions'] = missions_count

    # Check dataset sizes
    datasets_dir = os.path.join(WORK_DIR, 'datasets')
    if os.path.exists(datasets_dir):
        total_size = 0
        for root, dirs, files in os.walk(datasets_dir):
            for f in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, f))
                except:
                    pass

        size_mb = total_size / (1024 * 1024)
        size_gb = total_size / (1024 * 1024 * 1024)
        state['current_week']['progress']['dataset_growth_mb'] = round(size_mb, 2)
        state['current_month']['progress']['dataset_size_gb'] = round(size_gb, 2)
        updates['dataset_mb'] = round(size_mb, 2)
        updates['dataset_gb'] = round(size_gb, 2)

    # Check honeypot attacks
    attacks_file = os.path.join(WORK_DIR, 'datasets/honeypot/attacks.json')
    if os.path.exists(attacks_file):
        try:
            with open(attacks_file, 'r') as f:
                attacks = [line for line in f if line.strip()]
                state['current_week']['progress']['honeypot_attacks'] = len(attacks)
                updates['attacks'] = len(attacks)
        except:
            pass

    return updates

def generate_progress_report(state: Dict) -> str:
    """Generate progress report for current week and month"""
    report = []
    report.append("📊 <b>SYNEXS GOALS PROGRESS REPORT</b>")
    report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")

    # Weekly Progress
    week_start = datetime.fromisoformat(state['current_week']['start_date'])
    days_in_week = (datetime.now() - week_start).days
    week_completion = calculate_completion(state['current_week']['progress'], state['goals']['weekly'])

    report.append(f"📅 <b>WEEKLY GOALS</b> (Day {days_in_week}/7 - {week_completion:.1f}% Complete)")
    report.append("─" * 40)

    for goal_name, goal_config in state['goals']['weekly'].items():
        current = state['current_week']['progress'].get(goal_name, 0)
        target = goal_config['target']
        percentage = (current / target * 100) if target > 0 else 0
        status = "✅" if percentage >= 100 else "🔄" if percentage >= 50 else "⚠️"

        report.append(f"{status} {goal_config['description']}")
        report.append(f"   Progress: {current}/{target} {goal_config['unit']} ({percentage:.1f}%)")

    report.append("")

    # Monthly Progress
    month_start = datetime.fromisoformat(state['current_month']['start_date'])
    days_in_month = (datetime.now() - month_start).days
    month_completion = calculate_completion(state['current_month']['progress'], state['goals']['monthly'])

    report.append(f"📆 <b>MONTHLY GOALS</b> (Day {days_in_month}/30 - {month_completion:.1f}% Complete)")
    report.append("─" * 40)

    for goal_name, goal_config in state['goals']['monthly'].items():
        current = state['current_month']['progress'].get(goal_name, 0)
        target = goal_config['target']
        percentage = (current / target * 100) if target > 0 else 0
        status = "✅" if percentage >= 100 else "🔄" if percentage >= 50 else "⚠️"

        report.append(f"{status} {goal_config['description']}")
        report.append(f"   Progress: {current}/{target} {goal_config['unit']} ({percentage:.1f}%)")

    return "\n".join(report)

def generate_reminders(state: Dict) -> List[str]:
    """Generate action reminders based on goals"""
    reminders = []

    week_start = datetime.fromisoformat(state['current_week']['start_date'])
    days_in_week = (datetime.now() - week_start).days

    # Check weekly goals
    for goal_name, goal_config in state['goals']['weekly'].items():
        current = state['current_week']['progress'].get(goal_name, 0)
        target = goal_config['target']
        percentage = (current / target * 100) if target > 0 else 0

        if goal_config['priority'] == 'high' and percentage < 30 and days_in_week >= 3:
            reminders.append({
                'priority': 'high',
                'message': f"⚠️ <b>Weekly Goal Behind Schedule</b>: {goal_config['description']} at {percentage:.1f}%",
                'action': get_action_for_goal(goal_name)
            })

    # Check if training is needed
    if state['current_week']['progress'].get('gpu_training_runs', 0) == 0 and days_in_week >= 4:
        reminders.append({
            'priority': 'high',
            'message': "🎓 <b>Training Reminder</b>: No GPU training runs this week",
            'action': "Run: python3 synexs_gpu_trainer.py ./training_logs/batches"
        })

    return reminders

def get_action_for_goal(goal_name: str) -> str:
    """Get recommended action for a goal"""
    actions = {
        'missions_generated': 'Run: python3 synexs_phase1_runner.py --missions 1000',
        'gpu_training_runs': 'Run: python3 synexs_gpu_trainer.py ./training_logs/batches',
        'honeypot_attacks': 'Check honeypot_server.py is running and exposed',
        'dataset_growth_mb': 'Run: python3 dna_collector.py',
        'model_accuracy': 'Review training parameters and dataset quality'
    }
    return actions.get(goal_name, 'Review goal configuration')

def check_goals_and_remind():
    """Main function to check goals and send reminders"""
    state = load_goals_state()

    # Check for week/month rollover
    week_rolled = check_week_rollover(state)
    month_rolled = check_month_rollover(state)

    # Update progress from system
    updates = update_progress_from_system(state)

    # Generate report
    report = generate_progress_report(state)
    print(report)

    # Check if we should send reminders (once per day)
    should_remind = False
    if state['last_reminder']:
        last_reminder = datetime.fromisoformat(state['last_reminder'])
        hours_since = (datetime.now() - last_reminder).total_seconds() / 3600
        should_remind = hours_since >= 24
    else:
        should_remind = True

    # Generate and send reminders
    if should_remind:
        reminders = generate_reminders(state)

        if reminders:
            reminder_text = "\n\n".join([f"{r['message']}\n💡 Action: {r['action']}" for r in reminders])
            full_message = f"🔔 <b>Synexs Daily Reminders</b>\n\n{reminder_text}"
            send_telegram(full_message)
            state['last_reminder'] = datetime.now().isoformat()

    # Send weekly/monthly completion reports
    if week_rolled:
        week_completion = calculate_completion(
            state['history']['weekly'][-1]['progress'],
            state['goals']['weekly']
        )
        message = f"📅 <b>Weekly Goals Complete</b>\n\nCompletion: {week_completion:.1f}%"
        send_telegram(message)

    if month_rolled:
        month_completion = calculate_completion(
            state['history']['monthly'][-1]['progress'],
            state['goals']['monthly']
        )
        message = f"📆 <b>Monthly Goals Complete</b>\n\nCompletion: {month_completion:.1f}%"
        send_telegram(message)

    # Save state
    save_goals_state(state)

    return {
        'report': report,
        'updates': updates,
        'reminders_sent': should_remind
    }

def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--report':
            state = load_goals_state()
            update_progress_from_system(state)
            print(generate_progress_report(state))
            save_goals_state(state)
        elif sys.argv[1] == '--reset-week':
            state = load_goals_state()
            state['current_week'] = {
                'start_date': datetime.now().isoformat(),
                'progress': {goal: 0 for goal in state['goals']['weekly'].keys()}
            }
            save_goals_state(state)
            print("✅ Weekly goals reset")
        elif sys.argv[1] == '--reset-month':
            state = load_goals_state()
            state['current_month'] = {
                'start_date': datetime.now().isoformat(),
                'progress': {goal: 0 for goal in state['goals']['monthly'].keys()}
            }
            save_goals_state(state)
            print("✅ Monthly goals reset")
        else:
            print("Usage: synexs_goals_tracker.py [--report|--reset-week|--reset-month]")
    else:
        check_goals_and_remind()

if __name__ == "__main__":
    main()
