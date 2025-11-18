# Synexs v2.0 - Quick Start

## 🚀 Installation (Choose One Method)

### Method 1: Automated Install (Recommended)
```bash
chmod +x install_v2.sh
sudo ./install_v2.sh
systemctl start synexs-dashboard
```

### Method 2: Docker Compose
```bash
docker-compose up -d postgres redis dashboard
docker exec -it synexs_dashboard python3 db/migrate.py
```

### Method 3: Quick Start Script
```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

## 📊 Access Dashboard

Open your browser:
- **Map**: http://localhost:5000
- **Stats**: http://localhost:5000/stats
- **Attacks**: http://localhost:5000/attacks

## ✅ Verify Everything Works

```bash
# Check PostgreSQL
systemctl status postgresql

# Check listener.py (REQUIRED!)
python3 -m listener &

# Check dashboard
curl http://localhost:5000/api/health

# View logs
tail -f listener.log
journalctl -u synexs-dashboard -f
```

## 📁 Key Files Created

```
db/
├── models.py           # Database schema
├── database.py         # Connection manager
└── migrate.py          # Migration tool

dashboard/
├── app.py              # Flask app
└── templates/
    ├── index.html      # World map
    ├── stats.html      # Graphs
    └── attacks.html    # Table

listener.py             # UPDATED: PostgreSQL support
docker-compose.yml      # UPDATED: Added postgres/redis
requirements.txt        # UPDATED: New dependencies
```

## 🔧 Database Info

**Connection:**
- Host: localhost
- Port: 5432
- User: synexs
- Password: synexs_secure_pass_2024
- Database: synexs

**Tables:**
- `attacks` - Main attack data (replaces training_binary_v3.jsonl)
- `dashboard_stats` - Cached statistics (optional)

## 🎯 Features

### Live World Map
- Real-time attack locations
- Color-coded severity
- Clickable markers with details
- Auto-refresh every 10s

### Statistics Dashboard
- Total attacks & unique IPs
- Attacks per hour (24h graph)
- Top 10 countries
- Top 10 CVEs
- Recent attacks list

### Attack Table
- Last 100 attacks
- Pagination
- Searchable
- Auto-refresh

## 🔄 How It Works

```
Honeypot → Redis Queue → listener.py → Shodan/Nmap → PostgreSQL → Dashboard
                  ↓
            (Still works!)
              JSONL fallback
```

## ⚠️ Important Notes

1. **listener.py MUST be running** for new data
   ```bash
   python3 -m listener
   ```

2. **Compatible with existing setup** - No breaking changes

3. **Fallback to JSONL** if PostgreSQL unavailable

4. **Migration preserves** all existing training data

## 🐛 Troubleshooting

**Dashboard won't start:**
```bash
sudo systemctl start postgresql
python3 db/migrate.py
```

**No data showing:**
```bash
# Check listener is running
ps aux | grep listener

# Check database has data
sudo -u postgres psql -d synexs -c "SELECT COUNT(*) FROM attacks;"
```

**Redis connection failed:**
```bash
redis-server --daemonize yes
```

## 📚 Full Documentation

See `UPGRADE_GUIDE.md` for complete details.

## 🎨 Dark Hacker Theme

The dashboard features:
- Matrix-style green terminal theme
- Real-time attack visualization
- Leaflet.js world map
- Chart.js graphs
- Auto-refreshing data

Enjoy! 🔥
