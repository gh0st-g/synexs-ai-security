FROM python:3.11-slim

# Metadata
LABEL maintainer="Synexs AI Swarm"
LABEL description="Autonomous AI swarm with defensive training and self-healing"
LABEL version="3.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    requests==2.31.0 \
    anthropic==0.25.0 \
    dnspython==2.6.1 \
    scikit-learn==1.4.0 \
    joblib==1.3.2 \
    && pip cache purge

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/datasets/agents \
    /app/datasets/honeypot \
    /app/datasets/models \
    /app/logs \
    /app/comm_outbox \
    /app/inbox

# Set permissions
RUN chmod +x /app/*.py 2>/dev/null || true

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose ports
EXPOSE 5000 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV WORK_DIR=/app
ENV FLASK_APP=synexs_flask_dashboard.py

# Volume for persistent data
VOLUME ["/app/datasets", "/app/logs"]

# Entrypoint script for multi-process management
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["full"]
