#!/bin/bash
# Synexs v2.0 Installation Script
# Installs PostgreSQL, updates dependencies, and sets up dashboard

set -e

echo "=========================================="
echo "Synexs v2.0 Installation"
echo "PostgreSQL + Flask Dashboard Upgrade"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Please run as root or with sudo${NC}"
    exit 1
fi

# 1. Install PostgreSQL
echo -e "\n${GREEN}[1/6] Installing PostgreSQL...${NC}"
if command -v psql &> /dev/null; then
    echo "PostgreSQL already installed"
else
    apt-get update
    apt-get install -y postgresql postgresql-contrib
    systemctl start postgresql
    systemctl enable postgresql
fi

# 2. Create database and user
echo -e "\n${GREEN}[2/6] Creating database and user...${NC}"
sudo -u postgres psql -c "CREATE USER synexs WITH PASSWORD 'synexs_secure_pass_2024';" 2>/dev/null || echo "User already exists"
sudo -u postgres psql -c "CREATE DATABASE synexs OWNER synexs;" 2>/dev/null || echo "Database already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE synexs TO synexs;" 2>/dev/null

# 3. Install Python dependencies
echo -e "\n${GREEN}[3/6] Installing Python dependencies...${NC}"
pip3 install -r requirements.txt

# 4. Run database migration
echo -e "\n${GREEN}[4/6] Running database migration...${NC}"
python3 db/migrate.py

# 5. Set environment variables
echo -e "\n${GREEN}[5/6] Setting environment variables...${NC}"
cat > .env <<EOF
# PostgreSQL Configuration
POSTGRES_USER=synexs
POSTGRES_PASSWORD=synexs_secure_pass_2024
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=synexs
EOF

echo "Environment variables saved to .env"

# 6. Create systemd service for dashboard
echo -e "\n${GREEN}[6/6] Creating systemd service...${NC}"
cat > /etc/systemd/system/synexs-dashboard.service <<EOF
[Unit]
Description=Synexs Dashboard
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/synexs
EnvironmentFile=/root/synexs/.env
ExecStart=/usr/bin/python3 /root/synexs/dashboard/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo -e "\n${GREEN}=========================================="
echo "Installation Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Start the dashboard:"
echo "   systemctl start synexs-dashboard"
echo ""
echo "2. Enable auto-start on boot:"
echo "   systemctl enable synexs-dashboard"
echo ""
echo "3. Check status:"
echo "   systemctl status synexs-dashboard"
echo ""
echo "4. View dashboard:"
echo "   http://localhost:5000"
echo ""
echo "5. Make sure listener.py is running:"
echo "   python3 -m listener"
echo ""
echo "==========================================${NC}"
