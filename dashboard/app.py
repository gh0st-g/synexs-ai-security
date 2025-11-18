#!/usr/bin/env python3
"""
Synexs Real-Time Dashboard
Flask web application for monitoring honeypot attacks
"""

from flask import Flask, render_template, jsonify, request
from sqlalchemy import func, desc
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_db_session, test_connection
from db.models import Attack

app = Flask(__name__)
app.config['SECRET_KEY'] = 'synexs-dashboard-2024-secure-key'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# ==================== Routes ====================

@app.route('/')
def index():
    """
    Main dashboard - World map of attacking IPs
    """
    return render_template('index.html')


@app.route('/stats')
def stats():
    """
    Statistics page - Graphs and metrics
    """
    return render_template('stats.html')


@app.route('/attacks')
def attacks():
    """
    Attacks table - Last 100 attacks
    """
    return render_template('attacks.html')


# ==================== API Endpoints ====================

@app.route('/api/map-data')
def api_map_data():
    """
    Get attack data for world map (with lat/lon)
    Returns: [{ ip, lat, lon, country, vulns, severity }, ...]
    """
    try:
        db = get_db_session()

        # Get recent attacks with geolocation
        attacks = db.query(Attack).filter(
            Attack.latitude.isnot(None),
            Attack.longitude.isnot(None)
        ).order_by(desc(Attack.timestamp)).limit(500).all()

        data = []
        for attack in attacks:
            data.append({
                'ip': attack.ip,
                'lat': attack.latitude,
                'lon': attack.longitude,
                'country': attack.country or 'Unknown',
                'country_code': attack.country_code,
                'city': attack.city,
                'vulns': attack.vulns or [],
                'severity': attack.severity,
                'timestamp': attack.timestamp.isoformat() if attack.timestamp else None,
                'open_ports': attack.open_ports or []
            })

        db.close()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats-summary')
def api_stats_summary():
    """
    Get summary statistics
    Returns: { total_attacks, unique_ips, countries, top_cves, hourly_data }
    """
    try:
        db = get_db_session()

        # Total attacks
        total_attacks = db.query(func.count(Attack.id)).scalar()

        # Unique IPs
        unique_ips = db.query(func.count(func.distinct(Attack.ip))).scalar()

        # Top countries
        country_stats = db.query(
            Attack.country,
            func.count(Attack.id).label('count')
        ).filter(
            Attack.country.isnot(None)
        ).group_by(Attack.country).order_by(desc('count')).limit(10).all()

        top_countries = [{'country': c[0], 'count': c[1]} for c in country_stats]

        # Top CVEs (extract from vulns JSONB array)
        # Note: This is a simplified approach; for production, consider a separate CVE table
        all_attacks_with_vulns = db.query(Attack.vulns).filter(
            Attack.vulns.isnot(None)
        ).all()

        cve_counts = {}
        for (vulns,) in all_attacks_with_vulns:
            if vulns:
                for cve in vulns:
                    cve_counts[cve] = cve_counts.get(cve, 0) + 1

        top_cves = sorted(cve_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_cves = [{'cve': cve, 'count': count} for cve, count in top_cves]

        # Attacks per hour (last 24 hours)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        hourly_attacks = db.query(
            func.date_trunc('hour', Attack.timestamp).label('hour'),
            func.count(Attack.id).label('count')
        ).filter(
            Attack.timestamp >= twenty_four_hours_ago
        ).group_by('hour').order_by('hour').all()

        hourly_data = [
            {'hour': h[0].isoformat() if h[0] else None, 'count': h[1]}
            for h in hourly_attacks
        ]

        # Recent attacks (last 10)
        recent = db.query(Attack).order_by(desc(Attack.timestamp)).limit(10).all()
        recent_attacks = [
            {
                'ip': a.ip,
                'country': a.country,
                'vulns': a.vulns or [],
                'timestamp': a.timestamp.isoformat() if a.timestamp else None
            }
            for a in recent
        ]

        db.close()

        return jsonify({
            'total_attacks': total_attacks,
            'unique_ips': unique_ips,
            'top_countries': top_countries,
            'top_cves': top_cves,
            'hourly_data': hourly_data,
            'recent_attacks': recent_attacks
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/attacks-table')
def api_attacks_table():
    """
    Get paginated attacks table data
    Query params: page (default 1), per_page (default 100)
    """
    try:
        db = get_db_session()

        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 100))

        # Query with pagination
        attacks = db.query(Attack).order_by(
            desc(Attack.timestamp)
        ).limit(per_page).offset((page - 1) * per_page).all()

        total = db.query(func.count(Attack.id)).scalar()

        data = [attack.to_dict() for attack in attacks]

        db.close()

        return jsonify({
            'attacks': data,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def api_health():
    """
    Health check endpoint
    """
    db_status = test_connection()
    return jsonify({
        'status': 'healthy' if db_status else 'database_error',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'connected' if db_status else 'disconnected'
    })


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== Main ====================

if __name__ == '__main__':
    # Check database connection
    if not test_connection():
        print("ERROR: Cannot connect to database!")
        print("Make sure PostgreSQL is running and migrations are complete")
        print("Run: python3 db/migrate.py")
        sys.exit(1)

    print("=" * 60)
    print("Synexs Dashboard Starting...")
    print("=" * 60)
    print("Dashboard: http://localhost:5000")
    print("Stats: http://localhost:5000/stats")
    print("Attacks: http://localhost:5000/attacks")
    print("API Health: http://localhost:5000/api/health")
    print("=" * 60)

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
