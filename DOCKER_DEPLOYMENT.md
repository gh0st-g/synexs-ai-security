# 🐳 Docker Deployment Guide — Synexs Swarm v3.0

## 📦 What's Included

```
synexs/
├── Dockerfile                 # Multi-process container image
├── docker-compose.yml         # Orchestration config
├── docker-entrypoint.sh       # Process manager
├── .dockerignore             # Build optimization
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Build the Image

```bash
# Build the Docker image
docker build -t synexs-swarm:v3.0 .

# Or use docker-compose
docker-compose build
```

### 2. Run the Container

```bash
# Full stack (dashboard + swarm + honeypot)
docker-compose up -d

# Or run directly
docker run -d \
  --name synexs-swarm \
  -p 5000:5000 -p 8080:8080 \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/logs:/app/logs \
  synexs-swarm:v3.0
```

### 3. Check Status

```bash
# View logs
docker-compose logs -f

# Check health
curl http://localhost:5000/health

# View running processes
docker exec synexs-swarm ps aux
```

## 🎯 Run Modes

The entrypoint script supports different run modes:

### Full Stack (Default)
```bash
docker run synexs-swarm:v3.0 full
```
**Starts**: Dashboard + Swarm + Honeypot + Listener

### Dashboard Only
```bash
docker run -p 5000:5000 synexs-swarm:v3.0 dashboard
```
**Starts**: Flask dashboard only

### Swarm Only
```bash
docker run synexs-swarm:v3.0 swarm
```
**Starts**: AI swarm (no dashboard)

### Honeypot Only
```bash
docker run -p 8080:8080 synexs-swarm:v3.0 honeypot
```
**Starts**: Honeypot server only

### Core Loop Only
```bash
docker run synexs-swarm:v3.0 core
```
**Starts**: Core evolution loop only

## 🔧 Configuration

### Environment Variables

Set via `.env` file or docker-compose environment section:

```env
# API Keys
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id
CLAUDE_API_KEY=your_claude_api_key

# Swarm Configuration
CYCLE_INTERVAL=1800          # 30 minutes
MAX_PARALLEL_FILES=3         # Parallel file processing
DISK_MIN_FREE_GB=2          # Auto-cleanup threshold

# Flask
FLASK_ENV=production
```

### Volume Mounts

**Persistent Data**:
```yaml
volumes:
  - ./datasets:/app/datasets      # Training data, agents, attacks
  - ./logs:/app/logs              # Application logs
  - ./comm_outbox:/app/comm_outbox  # Communication outbox
  - ./inbox:/app/inbox            # Communication inbox
```

**Model Files**:
```yaml
volumes:
  - ./synexs_tag_model.pt:/app/synexs_tag_model.pt
  - ./vectorizer.joblib:/app/vectorizer.joblib
  - ./training_data.jsonl:/app/training_data.jsonl
```

## 📊 Resource Limits

Default limits (configured in docker-compose.yml):

```yaml
resources:
  limits:
    cpus: '2.0'        # Max 2 CPU cores
    memory: 4G         # Max 4GB RAM
  reservations:
    cpus: '0.5'        # Min 0.5 CPU cores
    memory: 512M       # Min 512MB RAM
```

Adjust based on your system:

```bash
# High-performance
docker run --cpus=4 --memory=8g synexs-swarm:v3.0

# Low-resource
docker run --cpus=1 --memory=1g synexs-swarm:v3.0
```

## 🏥 Health Checks

Built-in health monitoring:

```bash
# Check health status
docker inspect synexs-swarm | grep -A5 Health

# Manual health check
curl http://localhost:5000/health
```

**Health Check Config**:
- **Interval**: 60s
- **Timeout**: 10s
- **Retries**: 3
- **Start Period**: 30s

## 🔄 Self-Healing

The entrypoint script automatically:
- ✅ Monitors all processes every 30s
- ✅ Restarts crashed processes
- ✅ Logs restart events
- ✅ Handles graceful shutdown (SIGTERM)

## 📝 Logging

**Log Files**:
```bash
# Container logs
docker logs synexs-swarm -f

# Application logs (inside container)
docker exec synexs-swarm tail -f /app/logs/swarm.log
docker exec synexs-swarm tail -f /app/logs/honeypot.log

# Attack logs
docker exec synexs-swarm tail -f /app/datasets/honeypot/attacks.json
```

**Log Rotation**:
- Max size: 10MB per file
- Max files: 3 rotated files
- Driver: json-file

## 🛑 Stop & Cleanup

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Remove image
docker rmi synexs-swarm:v3.0

# Full cleanup
docker-compose down -v --rmi all
```

## 🔐 Security Best Practices

### 1. Don't Expose Ports Publicly
```yaml
ports:
  - "127.0.0.1:5000:5000"  # Localhost only
  - "127.0.0.1:8080:8080"
```

### 2. Use Secrets for API Keys
```bash
# Create .env file
echo "CLAUDE_API_KEY=your_key" > .env
chmod 600 .env

# Reference in docker-compose
env_file:
  - .env
```

### 3. Run as Non-Root (Optional)
```dockerfile
RUN useradd -m -u 1000 synexs
USER synexs
```

### 4. Read-Only Root Filesystem (Advanced)
```yaml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /app/logs
```

## 🚨 Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs synexs-swarm

# Check build logs
docker-compose build --no-cache

# Verify files exist
ls -la datasets/ logs/
```

### Port Already in Use
```bash
# Find process using port
sudo lsof -i :5000
sudo lsof -i :8080

# Kill process or change port
docker run -p 5001:5000 synexs-swarm:v3.0
```

### Out of Memory
```bash
# Increase memory limit
docker run --memory=8g synexs-swarm:v3.0

# Check memory usage
docker stats synexs-swarm
```

### Permission Denied
```bash
# Fix volume permissions
sudo chown -R 1000:1000 datasets/ logs/

# Or run as root (not recommended)
docker run --user root synexs-swarm:v3.0
```

## 📈 Monitoring

### Container Stats
```bash
# Real-time stats
docker stats synexs-swarm

# Resource usage
docker top synexs-swarm
```

### Process Monitoring
```bash
# Running processes
docker exec synexs-swarm ps aux | grep python

# Check specific process
docker exec synexs-swarm pgrep -af "ai_swarm_fixed"
```

### Disk Usage
```bash
# Container size
docker system df

# Volume sizes
du -sh datasets/ logs/
```

## 🎯 Production Deployment

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml synexs

# Scale services
docker service scale synexs_swarm=3
```

### Using Kubernetes
```bash
# Generate k8s manifests
kompose convert -f docker-compose.yml

# Deploy to k8s
kubectl apply -f synexs-swarm-deployment.yaml
```

## 📦 Multi-Architecture Build

Build for different platforms:

```bash
# Build for ARM64 (Raspberry Pi, Apple Silicon)
docker buildx build --platform linux/arm64 -t synexs-swarm:v3.0-arm64 .

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t synexs-swarm:v3.0 \
  --push .
```

## 🔄 Updates

```bash
# Rebuild with latest code
git pull
docker-compose build --no-cache
docker-compose up -d

# Or pull from registry
docker pull your-registry/synexs-swarm:latest
docker-compose up -d
```

## 📊 Benchmarks

**Startup Time**: ~30 seconds
**Memory Usage**: 800MB - 2GB
**CPU Usage**: 10-50% (2 cores)
**Disk I/O**: Moderate (log writes)
**Network**: Minimal (API calls only)

## ✅ Production Checklist

- [ ] Set environment variables in `.env`
- [ ] Configure volume mounts for persistent data
- [ ] Set appropriate resource limits
- [ ] Enable health checks
- [ ] Configure log rotation
- [ ] Set up monitoring/alerts
- [ ] Test graceful shutdown
- [ ] Verify auto-restart works
- [ ] Document recovery procedures
- [ ] Set up backups for datasets/

---

**Status**: 🚀 Production Ready
**Version**: 3.0
**Docker**: ✅ Tested
**Self-Healing**: ✅ Enabled
**Immortal**: 🛡️ Van-Proof
