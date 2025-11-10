# Synexs Phase 1 Training - Monitoring Guide

**Comprehensive guide for monitoring, managing, and troubleshooting Phase 1 training runs**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Using progress.sh Dashboard](#using-progresssh-dashboard)
3. [Checking Training Status](#checking-training-status)
4. [Reading Logs](#reading-logs)
5. [Estimating Completion Time](#estimating-completion-time)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Monitoring](#advanced-monitoring)

---

## Quick Start

### Starting a Training Run

```bash
# Quick test (10 missions) - completes in seconds
python3 synexs_phase1_runner.py --quick

# Standard training (1,000 missions) - completes in ~30 seconds
python3 synexs_phase1_runner.py --missions 1000

# Large-scale training (100K missions) - runs for ~45 minutes
nohup python3 synexs_phase1_runner.py --missions 100000 > train.out 2>&1 &
```

### Basic Monitoring

```bash
# Live dashboard (recommended)
./progress.sh training_logs

# Check if training process is running
ps aux | grep synexs_phase1_runner

# View real-time logs
tail -f training_logs/logs/training_*.log
```

---

## Using progress.sh Dashboard

### Overview

The `progress.sh` script provides a real-time dashboard showing:
- Current progress percentage and bar
- Mission counts (current/total)
- Processing rate (missions/sec)
- Estimated time to completion (ETA)
- Success/failure/abort statistics
- Training status

### Basic Usage

```bash
# Monitor default output directory
./progress.sh

# Monitor custom directory
./progress.sh ./my_training_logs

# Monitor with auto-refresh (default: 2 seconds)
./progress.sh training_logs
```

### Dashboard Output

```
═══════════════════════════════════════════════════════
              SYNEXS TRAINING MONITOR
═══════════════════════════════════════════════════════

Status: RUNNING
Progress: [████████████████████████░░░░░░] 65.0%

Mission Progress: 650 / 1000
Elapsed Time: 17.5 seconds
Processing Rate: 37.1 missions/sec
ETA: 9.4 seconds

Training Statistics:
  ✓ Successes: 372 (57.2%)
  ✗ Failures: 247 (38.0%)
  ⊗ Aborted: 31 (4.8%)

Last Updated: 2025-11-10 16:45:23

═══════════════════════════════════════════════════════
Monitoring: training_logs/progress.json
Press Ctrl+C to exit
```

### Understanding the Dashboard

| Element | Description |
|---------|-------------|
| **Status** | RUNNING, COMPLETED, or ERROR |
| **Progress Bar** | Visual representation of completion |
| **Mission Progress** | Current mission / Total missions |
| **Elapsed Time** | Time since training started |
| **Processing Rate** | Missions per second (sustained) |
| **ETA** | Estimated time to completion |
| **Successes** | Missions completed successfully |
| **Failures** | Missions that failed |
| **Aborted** | Missions that were aborted |

### Color Coding

- **Green (✓)**: Successes
- **Red (✗)**: Failures
- **Yellow (⊗)**: Aborted missions
- **Progress Bar**: █ = completed, ░ = remaining

---

## Checking Training Status

### Is Training Running?

```bash
# Check for running training process
ps aux | grep synexs_phase1_runner

# Example output:
# root  145232 15.9 18.0 3241268 364940 pts/4 R 16:36 1:19 python3 synexs_phase1_runner.py --missions 100000
```

**What to look for:**
- **PID** (145232): Process ID
- **CPU%** (15.9%): CPU usage - should be 10-30% for normal operation
- **MEM%** (18.0%): Memory usage - should be stable, typically 15-25%
- **Status** (R): Running status
  - `R` = Running (good)
  - `S` = Sleeping (waiting for I/O, normal)
  - `T` = Stopped (you paused it with Ctrl+Z)
  - `Z` = Zombie (crashed, needs cleanup)

### Quick Status Check

```bash
# Check progress file directly
cat training_logs/progress.json

# Example output:
{
  "timestamp": "2025-11-10T16:45:23.123456",
  "mission_current": 650,
  "mission_total": 1000,
  "progress_percent": 65.0,
  "elapsed_seconds": 17.5,
  "rate_missions_per_sec": 37.1,
  "eta_seconds": 9.4,
  "stats": {
    "success_count": 372,
    "failure_count": 247,
    "abort_count": 31
  },
  "status": "running"
}
```

### Check Checkpoint

```bash
# View current checkpoint
cat training_logs/checkpoint.json | python3 -m json.tool

# Key fields:
# - "mission_number": Last completed mission
# - "total_missions": Total missions to complete
# - "timestamp": When checkpoint was saved
```

---

## Reading Logs

### Log File Location

```bash
# Log files are in:
training_logs/logs/training_YYYYMMDD_HHMMSS.log
```

### Real-Time Log Viewing

```bash
# Follow log as it's written
tail -f training_logs/logs/training_*.log

# Last 100 lines
tail -n 100 training_logs/logs/training_*.log

# First 100 lines
head -n 100 training_logs/logs/training_*.log
```

### Filtering Logs

```bash
# Show only errors
grep ERROR training_logs/logs/training_*.log

# Show only warnings
grep WARNING training_logs/logs/training_*.log

# Show progress milestones
grep "Progress:" training_logs/logs/training_*.log

# Show checkpoint operations
grep "Checkpoint" training_logs/logs/training_*.log

# Show mission results
grep "Mission #" training_logs/logs/training_*.log
```

### Understanding Log Levels

| Level | Description | Example |
|-------|-------------|---------|
| **DEBUG** | Detailed information | Database queries, internal state |
| **INFO** | General information | Mission started, checkpoint saved |
| **WARNING** | Potential issues | High memory usage, slow performance |
| **ERROR** | Errors occurred | Mission failed, checkpoint corrupt |
| **CRITICAL** | Serious errors | System crash, data loss |

### Common Log Patterns

```bash
# Mission started
INFO - Mission #650 starting...

# Mission completed successfully
INFO - Mission #650 completed: SUCCESS

# Mission failed
INFO - Mission #650 completed: FAILURE

# Checkpoint saved
INFO - Checkpoint saved (Mission 650/1000, 65.0%)

# Progress milestone
INFO - Progress: 650/1000 (65.0%) - ETA: 9.4s
```

---

## Estimating Completion Time

### Using progress.sh

The dashboard automatically calculates ETA based on current processing rate.

### Manual Calculation

```bash
# 1. Get current progress
cat training_logs/progress.json | grep mission_current

# 2. Get processing rate
cat training_logs/progress.json | grep rate_missions_per_sec

# 3. Calculate remaining time
# remaining_missions = total - current
# eta_seconds = remaining_missions / rate
```

### Example Calculation

```
Total missions: 100,000
Current mission: 65,000
Processing rate: 37.1 missions/sec

Remaining: 100,000 - 65,000 = 35,000 missions
ETA: 35,000 / 37.1 = 943 seconds ≈ 16 minutes
```

### Expected Completion Times

| Missions | Typical Duration | Notes |
|----------|------------------|-------|
| 10 | ~0.2 seconds | Quick test |
| 100 | ~2 seconds | Development testing |
| 1,000 | ~30 seconds | Standard training |
| 10,000 | ~5 minutes | Large training |
| 100,000 | ~45 minutes | Production scale |

**Note:** Times assume ~37 missions/sec sustained rate. Actual rate may vary based on system performance.

---

## Troubleshooting

### Training Appears Stuck

**Symptoms:**
- Progress doesn't update for >1 minute
- CPU usage drops to 0%
- No new log entries

**Diagnosis:**
```bash
# Check if process is still running
ps aux | grep synexs_phase1_runner

# Check last log entry
tail -n 20 training_logs/logs/training_*.log

# Check system resources
free -h  # Memory
df -h    # Disk space
```

**Solutions:**
1. **Out of memory:** Kill process, increase system RAM
2. **Out of disk space:** Free up space, resume training
3. **Process crashed:** Check logs for errors, restart with same command (auto-resumes)

### High Memory Usage

**Symptoms:**
- Memory usage >80%
- System slowdown
- WARNING messages in logs

**Solutions:**
```bash
# Check memory usage
ps aux --sort=-%mem | head -10

# If too high, interrupt gracefully
kill -SIGINT <PID>

# Or use Ctrl+C if running in foreground

# Training will checkpoint and exit cleanly
# Resume with same command when memory is available
```

### Slow Processing Rate

**Expected:** ~37 missions/sec
**Acceptable:** >20 missions/sec

**If rate is <20 missions/sec:**

1. **Check CPU usage:**
   ```bash
   top -p <PID>
   ```
   - Should be 10-30% CPU
   - If <5%, system may be throttling

2. **Check disk I/O:**
   ```bash
   iotop -p <PID>
   ```
   - High I/O may slow processing
   - Consider using SSD

3. **Check for other processes:**
   ```bash
   top
   ```
   - Look for competing processes
   - Stop unnecessary services

### Training Won't Resume

**Symptoms:**
- "Checkpoint not found" message
- Starts from mission 0 after interruption

**Diagnosis:**
```bash
# Check if checkpoint exists
ls -la training_logs/checkpoint.json

# Try to read checkpoint
cat training_logs/checkpoint.json
```

**Solutions:**

1. **Checkpoint missing:**
   - Training will start fresh (expected behavior)
   - Previous progress is in mission logs

2. **Checkpoint corrupted:**
   ```bash
   # Validate JSON
   python3 -c "import json; json.load(open('training_logs/checkpoint.json'))"

   # If corrupted, remove it
   rm training_logs/checkpoint.json

   # Training will start fresh
   ```

3. **Wrong directory:**
   ```bash
   # Make sure you're using same --output directory
   python3 synexs_phase1_runner.py --missions 1000 --output training_logs
   ```

### Error: "No space left on device"

**Solutions:**

1. **Check disk space:**
   ```bash
   df -h
   ```

2. **Free up space:**
   ```bash
   # Remove old training runs
   rm -rf old_training_logs/

   # Compress old logs
   gzip training_logs/logs/training_*.log

   # Remove unnecessary files
   ```

3. **Clean up training artifacts:**
   ```bash
   # Remove old batches (can regenerate)
   rm training_logs/batches/*.pt

   # Archive old mission logs
   tar -czf missions_archive.tar.gz training_logs/missions/
   rm -rf training_logs/missions/
   ```

### Error: "Permission denied"

**Solutions:**

```bash
# Make progress.sh executable
chmod +x progress.sh

# Fix log directory permissions
chmod -R 755 training_logs/

# Run with appropriate permissions
sudo python3 synexs_phase1_runner.py --missions 1000
```

---

## Advanced Monitoring

### Monitoring Multiple Metrics

```bash
# Watch progress, CPU, and memory in one terminal
watch -n 2 '
  echo "=== Progress ===";
  cat training_logs/progress.json;
  echo "";
  echo "=== Resources ===";
  ps aux | grep synexs_phase1_runner | grep -v grep;
  echo "";
  echo "=== Disk ===";
  du -sh training_logs/;
'
```

### Tracking Performance Over Time

```bash
# Log performance metrics to file
while true; do
  echo "$(date +%s),$(cat training_logs/progress.json | grep rate_missions_per_sec | cut -d: -f2 | tr -d ' ,')" >> performance.csv
  sleep 10
done
```

### Remote Monitoring

```bash
# Set up SSH port forwarding if monitoring remotely
ssh -L 8080:localhost:8080 user@training-server

# Or use tmux for persistent sessions
tmux new -s training
python3 synexs_phase1_runner.py --missions 100000
# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t training
```

### Email Notifications

```bash
# Get notified when training completes
python3 synexs_phase1_runner.py --missions 100000 && \
  echo "Training complete!" | mail -s "Synexs Training Done" user@example.com
```

### Monitoring Checklist

**Before starting large runs (>10K missions):**
- [ ] Sufficient disk space (1GB+ per 10K missions)
- [ ] Sufficient RAM (512MB minimum, 1GB+ recommended)
- [ ] Not running on battery power
- [ ] progress.sh works in separate terminal
- [ ] Log directory writable

**During training:**
- [ ] Processing rate >20 missions/sec
- [ ] Memory usage stable (<80%)
- [ ] Disk space not decreasing rapidly
- [ ] No ERROR messages in logs

**After interruption:**
- [ ] Checkpoint file exists
- [ ] Checkpoint is valid JSON
- [ ] Resume with same --missions and --output arguments

---

## Storage Requirements

### Per-Mission Storage

| Component | Size | Location |
|-----------|------|----------|
| Mission log | ~2KB | `training_logs/missions/` |
| Training batch | ~1.5KB | `training_logs/batches/` |
| Checkpoint | ~1KB | `training_logs/checkpoint.json` |
| Progress | ~400B | `training_logs/progress.json` |

### Total Storage by Scale

| Missions | Missions | Batches | Logs | Total |
|----------|----------|---------|------|-------|
| 10 | 20KB | 15KB | 100KB | ~1MB |
| 100 | 200KB | 150KB | 500KB | ~5MB |
| 1,000 | 2MB | 1.5MB | 2MB | ~50MB |
| 10,000 | 20MB | 15MB | 10MB | ~500MB |
| 100,000 | 200MB | 150MB | 50MB | ~5GB |

**Note:** Add 20% buffer for checkpoints and overhead

---

## Performance Optimization Tips

### For Maximum Speed

1. **Use SSD storage** - 2-3x faster than HDD
2. **Disable unnecessary logging** - Use `--no-verbose` flag (if available)
3. **Increase checkpoint interval** - Automatic (adaptive)
4. **Run on dedicated system** - No competing processes

### For Long-Running Jobs

1. **Use nohup** - Survives disconnections
   ```bash
   nohup python3 synexs_phase1_runner.py --missions 100000 > train.out 2>&1 &
   ```

2. **Use tmux or screen** - Persistent terminal sessions
   ```bash
   tmux new -s training
   python3 synexs_phase1_runner.py --missions 100000
   ```

3. **Set up monitoring script** - Run progress.sh in separate terminal

4. **Enable email notifications** - Get alerted on completion

### For Reliability

1. **Regular backups** - Copy training_logs/ periodically
2. **Test resume functionality** - Interrupt and resume small runs
3. **Monitor disk space** - Set up alerts for <10% free
4. **Check logs regularly** - Look for WARNING/ERROR messages

---

## FAQ

### Q: Can I interrupt training safely?

**A:** Yes! Press Ctrl+C or send SIGINT. Training will:
1. Save current checkpoint
2. Update progress file
3. Exit gracefully
4. Resume from checkpoint when restarted

### Q: How do I resume after interruption?

**A:** Just run the same command:
```bash
python3 synexs_phase1_runner.py --missions 1000
```
It will automatically detect and resume from checkpoint.

### Q: Can I change mission count mid-training?

**A:** No. You must complete current run or start fresh with `--no-resume` flag.

### Q: What if checkpoint is corrupted?

**A:** Training will detect corruption and start fresh. Previous mission data is preserved in `training_logs/missions/`.

### Q: Can I run multiple training jobs simultaneously?

**A:** Yes, use different output directories:
```bash
python3 synexs_phase1_runner.py --missions 1000 --output ./train1 &
python3 synexs_phase1_runner.py --missions 1000 --output ./train2 &
```

### Q: How do I know if training succeeded?

**A:** Check `progress.json` status:
- `"status": "completed"` = Success
- `"status": "error"` = Failed
- `"status": "running"` = In progress

### Q: What's a good success rate?

**A:** ~50-60% is expected. Success rate depends on mission difficulty and agent capabilities.

### Q: Should I be concerned about FAILURE missions?

**A:** No. Failures are part of training data. Both successes and failures help the AI learn.

---

## Support

For issues not covered in this guide:

1. **Check logs:** `training_logs/logs/training_*.log`
2. **Review checkpoint:** `training_logs/checkpoint.json`
3. **Examine progress:** `training_logs/progress.json`
4. **Check documentation:** `OPTIMIZATION_REPORT.md`, `CHANGES_SUMMARY.md`

---

**Last Updated:** 2025-11-10
**Version:** 2.0
**Component:** Phase 1 Training Pipeline
