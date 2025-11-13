# Synexs Monitoring & Goals System

Complete guide for the enhanced monitoring, verification, and goals tracking system.

## 🎯 Overview

The Synexs monitoring system provides:
- **Comprehensive Health Checks** - Process, resource, and system monitoring
- **Goals Tracking** - Weekly and monthly dataset/training goals with reminders
- **Automated Alerts** - Telegram notifications for issues and reminders
- **Improvement Suggestions** - AI-driven recommendations for optimization

---

## 📊 Components

### 1. Comprehensive System Monitor
**File**: `synexs_comprehensive_monitor.py`

Monitors:
- ✅ Critical processes (listener, honeypot, ai_swarm, etc.)
- ✅ Docker container health
- ✅ System resources (CPU, memory, disk)
- ✅ Dataset sizes and growth
- ✅ Training data readiness
- ✅ GPU availability
- ✅ Log file sizes

**Usage**:
```bash
# Run manual check
python3 synexs_comprehensive_monitor.py

# Get JSON output
python3 synexs_comprehensive_monitor.py --json
```

**Runs automatically**: Every 4 hours via cron

### 2. Goals Tracker
**File**: `synexs_goals_tracker.py`

Tracks:
- 📅 **Weekly Goals**:
  - 5,000 training missions
  - 100MB dataset growth
  - 2+ GPU training runs
  - 1,000 honeypot attacks
  - 85%+ model accuracy

- 📆 **Monthly Goals**:
  - 20,000 total missions
  - 1GB dataset size
  - 4 model versions (weekly retraining)
  - 99% system uptime
  - 2+ documentation updates

**Usage**:
```bash
# Check current progress
python3 synexs_goals_tracker.py --report

# Reset weekly goals manually
python3 synexs_goals_tracker.py --reset-week

# Reset monthly goals manually
python3 synexs_goals_tracker.py --reset-month

# Run full check with reminders
python3 synexs_goals_tracker.py
```

**Runs automatically**:
- Daily at 9 AM (progress + reminders)
- Monday at 10 AM (weekly report)
- 1st of month at 8 AM (monthly report)

---

## 🔔 Automated Schedule

Current cron configuration:

| Task | Frequency | Purpose |
|------|-----------|---------|
| Process startup | @reboot | Launch critical processes |
| Log rotation | Hourly | Prevent log file bloat |
| Legacy health check | Every 6 hours | Basic monitoring |
| DNA collector | Every 30 mins | Gather training data |
| **Comprehensive monitor** | **Every 4 hours** | **Full system check** |
| **Goals tracker** | **Daily 9 AM** | **Progress + reminders** |
| **Weekly report** | **Monday 10 AM** | **Weekly summary** |
| **Auto GPU training** | **Sunday 2 AM** | **Weekly retraining** |
| **Monthly report** | **1st of month 8 AM** | **Monthly summary** |

---

## 📈 Current Status

Run this to see current system status:
```bash
python3 synexs_comprehensive_monitor.py
```

**Example Output**:
```
======================================================================
🔍 SYNEXS COMPREHENSIVE SYSTEM MONITOR
======================================================================

📋 PROCESSES: ✅ OK
   Running: 5

🐳 DOCKER: ✅ OK
   synexs-core: Up 42 hours

💻 RESOURCES: ✅ OK
   CPU: 45.2%
   Memory: 48.7%
   Disk: 65.0%

📊 DATASETS: ✅ OK
   Total Size: 2263.2MB
   training_logs: 9376 batches ready

🎓 TRAINING: ✅ READY
   GPU Available: Yes (NVIDIA RTX 3090)
   Training Batches: 9376

💡 RECOMMENDATIONS:
   • 🎓 9376 training batches ready! Run: python3 synexs_gpu_trainer.py
   • 📊 Dataset growing well. Consider scaling to 10K missions
```

---

## 🎯 Goals & Progress

Check your goals progress:
```bash
python3 synexs_goals_tracker.py --report
```

**Example Output**:
```
📊 SYNEXS GOALS PROGRESS REPORT
📅 Generated: 2025-11-13 09:00

📅 WEEKLY GOALS (Day 3/7 - 72.5% Complete)
────────────────────────────────────────
✅ Generate 5,000 training missions per week
   Progress: 6,240/5000 missions (124.8%)

🔄 Grow datasets by 100MB per week
   Progress: 78/100 MB (78.0%)

✅ Run GPU training at least 2 times per week
   Progress: 2/2 runs (100.0%)

🔄 Collect 1,000 honeypot attacks per week
   Progress: 456/1000 attacks (45.6%)

⚠️ Maintain model accuracy above 85%
   Progress: 0/85 % (0.0%)
```

---

## 🚨 Alerts & Notifications

All alerts are sent via **Telegram** to your configured chat.

**Alert Types**:
1. **Process Issues** - Missing or excess processes
2. **Resource Warnings** - High CPU/memory/disk usage
3. **Docker Problems** - Container restarts or failures
4. **Dataset Stagnation** - No data growth detected
5. **Training Reminders** - When GPU training is overdue
6. **Goal Reminders** - Weekly/monthly goal progress

**Configuration**:
Edit `.env` file:
```bash
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## 🔧 Troubleshooting

### Issue: Missing Processes

**Symptom**: Monitor reports processes as missing

**Fix**:
```bash
# Check what's actually running
ps aux | grep python3

# Restart specific process
cd /root/synexs
nohup python3 honeypot_server.py > /dev/null 2>&1 &
```

### Issue: Docker Container Restarting

**Symptom**: `synexs-swarm` or `claude-swarm` constantly restarting

**Fix**:
```bash
# Check logs
docker logs synexs-swarm

# Rebuild containers (after fixing Dockerfile)
cd /root/synexs
docker-compose down
docker-compose build
docker-compose up -d
```

### Issue: High CPU Usage

**Symptom**: CPU consistently above 80%

**Investigate**:
```bash
# Check which process is using CPU
top -bn1 | head -20

# If it's listener.py, check for process leaks
ps aux | grep listener.py | wc -l

# Kill excess processes if needed
pkill -9 -f listener.py
```

### Issue: Goals Not Updating

**Symptom**: Goals tracker shows 0 progress

**Fix**:
```bash
# Check state file
cat /root/synexs/.goals_tracker.json

# Reset if corrupted
python3 synexs_goals_tracker.py --reset-week
```

---

## 📊 Recommendations System

The monitor provides intelligent suggestions based on:

1. **Dataset Size**
   - Small datasets → Increase collection frequency
   - Large datasets → Ready for serious training

2. **Training Status**
   - No recent training → Schedule GPU run
   - Training data ready → Run trainer now

3. **Resource Usage**
   - High CPU → Optimize or distribute workload
   - High memory → Review process efficiency
   - High disk → Clean old logs/datasets

4. **Process Health**
   - Missing processes → Auto-restart recommendations
   - Excess processes → Process leak warnings

---

## 🎓 Training Workflow

**Automated Weekly Training** (Sunday 2 AM):
```bash
# Happens automatically via cron
python3 synexs_gpu_trainer.py ./training_logs/batches
```

**Manual Training**:
```bash
# 1. Check if data is ready
python3 synexs_comprehensive_monitor.py

# 2. Generate more data if needed
python3 synexs_phase1_runner.py --missions 1000

# 3. Run GPU training
python3 synexs_gpu_trainer.py ./training_logs/batches

# 4. Check results
ls -lh *.pt
```

---

## 📁 Generated Files

**State Files**:
- `.monitor_state.json` - Monitor state and dataset sizes
- `.goals_tracker.json` - Goals progress and history

**Log Files**:
- `comprehensive_monitor.log` - Monitor execution logs
- `goals_tracker.log` - Goals tracking logs
- `training_auto.log` - Automated training logs

**Don't delete** these files - they track your progress!

---

## 🔄 Manual Commands

### Quick Health Check
```bash
python3 synexs_comprehensive_monitor.py
```

### Goals Report
```bash
python3 synexs_goals_tracker.py --report
```

### Force Training
```bash
python3 synexs_gpu_trainer.py ./training_logs/batches
```

### Generate More Data
```bash
# Quick test (10 missions)
python3 synexs_phase1_runner.py --quick

# Production batch (1000 missions)
python3 synexs_phase1_runner.py --missions 1000

# Large batch (10,000 missions)
python3 synexs_phase1_runner.py --missions 10000
```

---

## 📞 Support

**Logs Location**: `/root/synexs/*.log`

**Check Cron Status**:
```bash
crontab -l  # View scheduled tasks
tail -f /var/log/syslog | grep CRON  # Watch cron execution
```

**Manual Telegram Test**:
```bash
python3 -c "
import requests
import os
token = os.getenv('TELEGRAM_TOKEN', '8204790720:AAEFxHurgJGIigQh0MtOlUbxX46PU8A2rA8')
chat = os.getenv('TELEGRAM_CHAT_ID', '1749138955')
url = f'https://api.telegram.org/bot{token}/sendMessage'
data = {'chat_id': chat, 'text': '✅ Synexs monitoring test'}
print(requests.post(url, data=data).json())
"
```

---

## ✅ Checklist

Daily:
- [ ] Check Telegram for alerts
- [ ] Review goals progress (if needed)

Weekly:
- [ ] Review Monday morning report
- [ ] Verify training completed Sunday
- [ ] Check dataset growth

Monthly:
- [ ] Review monthly goals report
- [ ] Analyze training improvements
- [ ] Update documentation

---

**Last Updated**: 2025-11-13
**Version**: 1.0
