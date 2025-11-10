#!/bin/bash
# Synexs Swarm - Docker Quick Commands

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        🐳 SYNEXS SWARM - DOCKER QUICK REFERENCE           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cat << 'HELP'
📦 BUILD
  docker build -t synexs-swarm:v3.0 .
  docker-compose build

🚀 RUN
  docker-compose up -d                    # Full stack
  docker-compose up -d --scale swarm=3    # Scale to 3 instances

🛑 STOP
  docker-compose down                     # Stop all
  docker-compose down -v                  # Stop + remove volumes

📊 MONITOR
  docker-compose logs -f                  # Live logs
  docker stats synexs-swarm               # Resource usage
  curl http://localhost:5000/health       # Health check

🔧 DEBUG
  docker exec -it synexs-swarm bash       # Shell access
  docker exec synexs-swarm ps aux         # Running processes
  docker logs synexs-swarm --tail 100     # Last 100 log lines

🔄 UPDATE
  git pull && docker-compose build --no-cache && docker-compose up -d

🧹 CLEANUP
  docker system prune -a                  # Clean everything
  docker volume prune                     # Clean volumes

📍 MODES
  docker run synexs-swarm:v3.0 full       # All services
  docker run synexs-swarm:v3.0 dashboard  # Dashboard only
  docker run synexs-swarm:v3.0 swarm      # Swarm only
  docker run synexs-swarm:v3.0 honeypot   # Honeypot only

🔐 SECURITY
  # Use .env file for secrets
  echo "CLAUDE_API_KEY=sk-xxx" > .env
  chmod 600 .env

  # Localhost only
  ports:
    - "127.0.0.1:5000:5000"

🎯 PRODUCTION
  # Set resource limits
  docker run --cpus=2 --memory=4g synexs-swarm:v3.0

  # Auto-restart
  docker run --restart=unless-stopped synexs-swarm:v3.0

  # Health checks
  docker inspect synexs-swarm | grep -A5 Health

HELP

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Ready to deploy!"
echo "═══════════════════════════════════════════════════════════"
