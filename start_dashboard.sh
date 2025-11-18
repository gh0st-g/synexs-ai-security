#!/bin/bash
# Quick start script for Synexs Dashboard

echo "Starting Synexs Dashboard..."

# Check if PostgreSQL is running
if ! systemctl is-active --quiet postgresql; then
    echo "Starting PostgreSQL..."
    systemctl start postgresql
fi

# Check if Redis is running
if ! systemctl is-active --quiet redis-server 2>/dev/null; then
    if ! pgrep -x "redis-server" > /dev/null; then
        echo "Starting Redis..."
        redis-server --daemonize yes
    fi
fi

# Set environment variables
export POSTGRES_USER=${POSTGRES_USER:-synexs}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-synexs_secure_pass_2024}
export POSTGRES_HOST=${POSTGRES_HOST:-localhost}
export POSTGRES_PORT=${POSTGRES_PORT:-5432}
export POSTGRES_DB=${POSTGRES_DB:-synexs}

# Start dashboard
cd "$(dirname "$0")"
python3 dashboard/app.py
