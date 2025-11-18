# Synexs v2.0 Upgrade Guide

## What's New

This upgrade adds:

1. **PostgreSQL Database** - Replaces `training_binary_v3.jsonl` with a real database
2. **Flask Dashboard** - Real-time web interface with:
   - Live world map of attacking IPs (Leaflet.js)
   - Statistics graphs (attacks/hour, top CVEs, top countries)
   - Attack table (last 100 attacks with pagination)
   - Dark hacker theme
   - Auto-refresh every 10 seconds

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Honeypot/      │────>│    Redis     │────>│ listener.py  │
│  send_agent.py  │     │ (queue)      │     │              │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                                     │ Enriches with
                                                     │ Shodan/Nmap
                                                     ▼
                                              ┌──────────────┐
                                              │ PostgreSQL   │
                                              │   Database   │
                                              └──────┬───────┘
                                                     │
                                                     │ Queries
                                                     ▼
                                              ┌──────────────┐
                                              │   Flask      │
                                              │  Dashboard   │
                                              └──────────────┘
                                              http://localhost:5000
```

## Folder Structure

```
/root/synexs/
├── db/
│   ├── __init__.py
│   ├── models.py           # SQLAlchemy models (Attack, DashboardStats)
│   ├── database.py         # Database connection manager
│   └── migrate.py          # Migration script (jsonl → postgres)
│
├── dashboard/
│   ├── app.py              # Flask application
│   ├── templates/
│   │   ├── base.html       # Base template with dark theme
│   │   ├── index.html      # Live world map
│   │   ├── stats.html      # Statistics graphs
│   │   └── attacks.html    # Attack table
│   └── static/
│       ├── css/
│       └── js/
│
├── utils/
│   ├── telegram_notify.py
│   └── external_intel.py   # Updated with geolocation
│
├── listener.py             # UPDATED: Now writes to PostgreSQL
├── docker-compose.yml      # UPDATED: Added postgres, redis, dashboard
├── requirements.txt        # UPDATED: Added SQLAlchemy, psycopg2, Flask deps
├── install_v2.sh           # Installation script
├── start_dashboard.sh      # Quick start script
└── UPGRADE_GUIDE.md        # This file
```

## Installation Methods

### Option 1: Docker Compose (Recommended)

1. **Start services:**
   ```bash
   docker-compose up -d postgres redis dashboard
   ```

2. **Run migration:**
   ```bash
   docker exec -it synexs_dashboard python3 db/migrate.py
   ```

3. **Access dashboard:**
   ```
   http://localhost:5000
   ```

### Option 2: Native Installation

1. **Run installation script:**
   ```bash
   chmod +x install_v2.sh
   sudo ./install_v2.sh
   ```

2. **Start dashboard:**
   ```bash
   systemctl start synexs-dashboard
   systemctl enable synexs-dashboard  # Auto-start on boot
   ```

3. **Access dashboard:**
   ```
   http://localhost:5000
   ```

### Option 3: Manual Installation

1. **Install PostgreSQL:**
   ```bash
   sudo apt-get update
   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql
   ```

2. **Create database:**
   ```bash
   sudo -u postgres psql
   CREATE USER synexs WITH PASSWORD 'synexs_secure_pass_2024';
   CREATE DATABASE synexs OWNER synexs;
   GRANT ALL PRIVILEGES ON DATABASE synexs TO synexs;
   \q
   ```

3. **Install Python dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Run migration:**
   ```bash
   python3 db/migrate.py
   ```

5. **Start dashboard:**
   ```bash
   chmod +x start_dashboard.sh
   ./start_dashboard.sh
   ```

## Database Schema

### `attacks` Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| ip | VARCHAR(45) | Attacker IP (IPv4/IPv6) |
| timestamp | DATETIME | Attack timestamp |
| vulns | JSONB | Array of CVEs |
| open_ports | JSONB | Array of open ports |
| country | VARCHAR(100) | Country name |
| country_code | VARCHAR(3) | ISO country code |
| city | VARCHAR(100) | City name |
| latitude | FLOAT | Latitude (for map) |
| longitude | FLOAT | Longitude (for map) |
| actions | JSONB | Binary protocol actions |
| binary_input | TEXT | Base64 encoded binary |
| protocol | VARCHAR(10) | Protocol version (v3) |
| format | VARCHAR(20) | Encoding format (base64) |
| instruction | TEXT | Training instruction |
| output | TEXT | Training output |
| source | VARCHAR(50) | Data source (shodan_nmap) |
| org | VARCHAR(255) | Organization |
| isp | VARCHAR(255) | ISP |
| asn | VARCHAR(50) | ASN |
| is_threat | BOOLEAN | Threat flag |
| severity | VARCHAR(20) | low, medium, high, critical |
| created_at | DATETIME | Record creation time |
| updated_at | DATETIME | Last update time |

## Dashboard Routes

| Route | Description |
|-------|-------------|
| `/` | Live world map of attacks |
| `/stats` | Statistics and graphs |
| `/attacks` | Attack table (last 100) |
| `/api/map-data` | JSON: Attack geolocation data |
| `/api/stats-summary` | JSON: Summary statistics |
| `/api/attacks-table` | JSON: Paginated attack table |
| `/api/health` | JSON: Health check |

## Configuration

### Environment Variables

Create a `.env` file:

```bash
POSTGRES_USER=synexs
POSTGRES_PASSWORD=synexs_secure_pass_2024
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=synexs
```

### Database Connection

Default connection string:
```
postgresql://synexs:synexs_secure_pass_2024@localhost:5432/synexs
```

## How listener.py Works Now

1. **Reads from Redis queue** (`agent_tasks`)
2. **Enriches IP** with Shodan/Nmap (includes geolocation now)
3. **Writes to PostgreSQL** (preferred) or falls back to JSONL
4. **Sends Telegram alert**

### Fallback Behavior

If PostgreSQL is unavailable, listener.py automatically falls back to writing JSONL files. No data loss!

## Migration Script

The migration script (`db/migrate.py`) does:

1. Tests database connection
2. Creates all tables
3. Creates performance indexes
4. Migrates existing `training_binary_v3.jsonl` to PostgreSQL
5. Preserves all training data

Run manually:
```bash
python3 db/migrate.py
```

## Dashboard Features

### 1. World Map (`/`)
- Real-time attack locations
- Color-coded severity (red = critical, orange = high, blue = medium)
- Popup with IP, country, CVEs, ports
- Live stats: active threats, countries, CVEs

### 2. Statistics (`/stats`)
- Total attacks counter
- Unique IPs counter
- Attacks per hour graph (last 24h)
- Top 10 countries (bar chart)
- Top 10 CVEs table
- Recent attacks list

### 3. Attack Table (`/attacks`)
- Last 100 attacks
- Pagination support
- Columns: ID, timestamp, IP, country, CVEs, ports, severity
- Auto-refresh every 10 seconds

## Running Everything

### Start Order

1. **PostgreSQL** (if not using Docker)
   ```bash
   systemctl start postgresql
   ```

2. **Redis** (if not using Docker)
   ```bash
   redis-server --daemonize yes
   ```

3. **listener.py** (required!)
   ```bash
   python3 -m listener
   ```

4. **Dashboard**
   ```bash
   python3 dashboard/app.py
   # OR
   systemctl start synexs-dashboard
   ```

### Docker Compose Method

Start everything at once:
```bash
docker-compose up -d
```

Services:
- `postgres` → Port 5432
- `redis` → Port 6379
- `dashboard` → Port 5000

## Compatibility

✅ **Fully compatible** with existing setup:
- listener.py still runs as `python3 -m listener`
- Falls back to JSONL if database unavailable
- Existing training data can be migrated
- No changes to honeypot/send_agent.py needed

## Testing

1. **Test database connection:**
   ```bash
   python3 -c "from db.database import test_connection; print(test_connection())"
   ```

2. **Test migration:**
   ```bash
   python3 db/migrate.py
   ```

3. **Test dashboard:**
   ```bash
   curl http://localhost:5000/api/health
   ```

4. **Send test attack:**
   ```bash
   python3 -c "import redis, json; r = redis.Redis(); r.rpush('agent_tasks', json.dumps({'source_ip': '8.8.8.8'}))"
   ```

## Troubleshooting

### Dashboard won't start

Check PostgreSQL is running:
```bash
systemctl status postgresql
```

Check connection:
```bash
python3 db/migrate.py
```

### No attacks on map

1. Check listener.py is running:
   ```bash
   ps aux | grep listener
   ```

2. Check database has data:
   ```bash
   sudo -u postgres psql -d synexs -c "SELECT COUNT(*) FROM attacks;"
   ```

3. Check Redis queue:
   ```bash
   redis-cli LLEN agent_tasks
   ```

### Migration fails

Make sure PostgreSQL is running and accessible:
```bash
sudo -u postgres psql -l
```

## Security Notes

1. **Change default password** in production:
   ```bash
   # In .env and docker-compose.yml
   POSTGRES_PASSWORD=your_secure_password_here
   ```

2. **Firewall dashboard** if exposed:
   ```bash
   # Only allow localhost
   ufw allow from 127.0.0.1 to any port 5000
   ```

3. **Use HTTPS** in production (add nginx reverse proxy)

## Support

If you encounter issues:

1. Check logs:
   ```bash
   # Dashboard logs
   journalctl -u synexs-dashboard -f

   # Listener logs
   tail -f listener.log

   # PostgreSQL logs
   tail -f /var/log/postgresql/postgresql-*.log
   ```

2. Check service status:
   ```bash
   systemctl status synexs-dashboard
   systemctl status postgresql
   systemctl status redis-server
   ```

## Next Steps

After installation:

1. Visit dashboard: http://localhost:5000
2. Ensure listener.py is running
3. Monitor attack map in real-time
4. Set up Telegram alerts (if not already configured)
5. Configure auto-start for all services

## Performance

- Database: Handles 100k+ attacks easily
- Dashboard: Auto-refresh every 10 seconds
- Indexes on: IP, timestamp, country_code, severity
- Connection pooling: 10 connections, 20 max overflow

Enjoy your upgraded Synexs v2.0! 🚀
