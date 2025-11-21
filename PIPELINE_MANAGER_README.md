# SYNEXS Pipeline Manager

Automated process management and monitoring system for the SYNEXS pipeline.

## Quick Start

The pipeline manager is now available as a simple command:

```bash
synexs status    # Check pipeline status
synexs start     # Start all processes
synexs stop      # Stop all processes
synexs restart   # Restart all processes
```

## Features

✅ **Automatic Process Management**
- Start/stop all pipeline processes with one command
- Individual process health checks
- Auto-restart failed processes in monitor mode

✅ **Configuration-Based**
- Easy to add/remove processes
- Centralized configuration in `pipeline_config.json`
- Support for custom Python interpreters and working directories

✅ **Health Monitoring**
- HTTP endpoint health checks
- Redis service monitoring
- Real-time status reporting

✅ **Auto-Start on Reboot**
- Systemd service integration
- Automatic startup after system reboot
- Continuous monitoring service available

## Commands

### Status Check
```bash
synexs status
```
Shows the current status of all processes and services with health checks.

### Start All Processes
```bash
synexs start
```
Starts all enabled processes defined in the configuration.

### Stop All Processes
```bash
synexs stop
```
Gracefully stops all running processes.

### Restart All Processes
```bash
synexs restart
```
Stops and restarts all processes.

### Add New Process
```bash
synexs add
```
Interactive wizard to add a new process to the configuration.

### Remove Process
```bash
synexs remove
```
Interactive menu to remove a process from the configuration.

### List Processes
```bash
synexs list
```
Lists all configured processes with their settings.

### Continuous Monitoring
```bash
synexs monitor
```
Runs continuous monitoring that checks status every 30 seconds and auto-restarts failed processes.

## Configuration

Configuration file: `/root/synexs/pipeline_config.json`

### Process Configuration Structure

```json
{
  "name": "process_name",
  "script": "/path/to/script.py",
  "python": "/usr/bin/python3",
  "working_dir": "/root/synexs",
  "args": [],
  "enabled": true,
  "health_check": {
    "type": "http",
    "url": "http://localhost:8080/health",
    "expect_key": "status",
    "expect_value": "healthy"
  }
}
```

### Current Processes

1. **synexs_core_orchestrator** - Core orchestration engine
2. **listener** - Redis queue listener with HTTP health endpoint
3. **ai_swarm_fixed** - AI swarm coordination
4. **honeypot_server** - Hybrid WAF + AI honeypot with health endpoint

### Adding a New Process

**Option 1: Interactive**
```bash
synexs add
```

**Option 2: Manual Edit**
Edit `/root/synexs/pipeline_config.json` and add your process to the `processes` array.

### Removing a Process

**Option 1: Interactive**
```bash
synexs remove
```

**Option 2: Manual Edit**
Edit `/root/synexs/pipeline_config.json` and remove the process from the `processes` array.

## Auto-Start on Reboot

### Enable Auto-Start
```bash
sudo systemctl enable synexs-pipeline.service
```

### Enable Continuous Monitoring (Optional)
```bash
sudo systemctl enable synexs-monitor.service
sudo systemctl start synexs-monitor.service
```

### Check Service Status
```bash
sudo systemctl status synexs-pipeline
sudo systemctl status synexs-monitor
```

### Disable Auto-Start
```bash
sudo systemctl disable synexs-pipeline.service
sudo systemctl disable synexs-monitor.service
```

## Logs

- Process logs: `/root/synexs/logs/<process_name>.log`
- PID files: `/root/synexs/pids/<process_name>.pid`

### View Process Logs
```bash
tail -f /root/synexs/logs/honeypot_server.log
tail -f /root/synexs/logs/listener.log
```

### View All Recent Logs
```bash
tail -f /root/synexs/logs/*.log
```

## Systemd Integration

Two systemd services are available:

### 1. synexs-pipeline.service
- Runs once at boot to start all processes
- Use: `sudo systemctl start synexs-pipeline`

### 2. synexs-monitor.service
- Continuously monitors and auto-restarts failed processes
- Checks every 30 seconds
- Use: `sudo systemctl start synexs-monitor`

### Manual Service Control
```bash
# Start pipeline on boot
sudo systemctl enable synexs-pipeline

# Start continuous monitoring
sudo systemctl enable synexs-monitor
sudo systemctl start synexs-monitor

# Check status
sudo systemctl status synexs-pipeline
sudo systemctl status synexs-monitor

# View logs
sudo journalctl -u synexs-pipeline -f
sudo journalctl -u synexs-monitor -f
```

## Troubleshooting

### Process Won't Start
1. Check the logs: `tail -50 /root/synexs/logs/<process_name>.log`
2. Verify the script path is correct in `pipeline_config.json`
3. Ensure the Python interpreter path is correct
4. Check file permissions

### Health Check Failing
1. Verify the endpoint is accessible: `curl http://localhost:8080/health`
2. Check if the expected JSON keys/values are correct
3. Review the health check configuration in `pipeline_config.json`

### Service Not Starting on Reboot
1. Check if systemd service is enabled: `systemctl is-enabled synexs-pipeline`
2. View service logs: `journalctl -u synexs-pipeline -b`
3. Verify service file: `systemctl cat synexs-pipeline`

## Examples

### Quick Health Check After Reboot
```bash
synexs status
```

### Start Everything Manually
```bash
synexs start
```

### Add a New Custom Process
```bash
synexs add
# Follow the prompts:
# Process name: my_custom_service
# Script path: /root/synexs/my_service.py
# Python path: /usr/bin/python3
# Working directory: /root/synexs
# Arguments: --verbose
# Add HTTP health check? n
```

### Monitor Continuously (Auto-Restart)
```bash
synexs monitor
# Press Ctrl+C to stop monitoring
```

### Enable Auto-Start and Monitoring
```bash
sudo systemctl enable synexs-pipeline
sudo systemctl enable synexs-monitor
sudo systemctl start synexs-monitor
```

Now the pipeline will:
- ✅ Auto-start all processes on reboot
- ✅ Continuously monitor health every 30 seconds
- ✅ Auto-restart any failed processes

## Files Created

- `/root/synexs/pipeline_config.json` - Process configuration
- `/root/synexs/pipeline_manager.py` - Main manager script
- `/usr/local/bin/synexs` - Quick access command
- `/etc/systemd/system/synexs-pipeline.service` - Auto-start service
- `/etc/systemd/system/synexs-monitor.service` - Monitoring service
- `/root/synexs/logs/` - Process logs directory
- `/root/synexs/pids/` - PID files directory
