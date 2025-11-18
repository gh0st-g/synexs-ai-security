"""
Synexs PostgreSQL Database Models
SQLAlchemy ORM models for attack data storage
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import json

Base = declarative_base()


class Attack(Base):
    """
    Attack records from honeypot
    Replaces training_binary_v3.jsonl with structured database
    """
    __tablename__ = 'attacks'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Core attack data
    ip = Column(String(45), nullable=False, index=True)  # Support IPv6
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Intel data (from Shodan/Nmap)
    vulns = Column(JSONB, default=list)  # List of CVEs
    open_ports = Column(JSONB, default=list)  # List of open ports

    # Geolocation (for map)
    country = Column(String(100), nullable=True)
    country_code = Column(String(3), nullable=True, index=True)
    city = Column(String(100), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Binary protocol data
    actions = Column(JSONB, default=list)  # e.g., ["SCAN", "VULN", "EXPLOIT"]
    binary_input = Column(Text, nullable=True)  # base64 encoded binary
    protocol = Column(String(10), default='v3')
    format = Column(String(20), default='base64')

    # Training data fields (preserves AI training format)
    instruction = Column(Text, nullable=True)
    output = Column(Text, nullable=True)
    source = Column(String(50), default='shodan_nmap', index=True)

    # Additional metadata
    org = Column(String(255), nullable=True)  # Organization from Shodan
    isp = Column(String(255), nullable=True)
    asn = Column(String(50), nullable=True)

    # Analysis flags
    is_threat = Column(Boolean, default=True)
    severity = Column(String(20), default='medium')  # low, medium, high, critical

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Attack(id={self.id}, ip={self.ip}, vulns={len(self.vulns or [])})>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'ip': self.ip,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'vulns': self.vulns or [],
            'open_ports': self.open_ports or [],
            'country': self.country,
            'country_code': self.country_code,
            'city': self.city,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'actions': self.actions or [],
            'binary_input': self.binary_input,
            'protocol': self.protocol,
            'format': self.format,
            'instruction': self.instruction,
            'output': self.output,
            'source': self.source,
            'org': self.org,
            'isp': self.isp,
            'asn': self.asn,
            'is_threat': self.is_threat,
            'severity': self.severity,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def to_training_format(self):
        """
        Convert to JSONL training format (compatible with original system)
        This allows AI training to continue working
        """
        return {
            "instruction": self.instruction,
            "input": f"binary:{self.binary_input}" if self.binary_input else "",
            "output": self.output,
            "actions": self.actions or [],
            "protocol": self.protocol,
            "format": self.format,
            "source": self.source,
            "timestamp": int(self.timestamp.timestamp()) if self.timestamp else None,
            "ip": self.ip,
            "vulns": self.vulns or [],
            "open_ports": self.open_ports or []
        }


class DashboardStats(Base):
    """
    Aggregated statistics for dashboard (optional caching table)
    """
    __tablename__ = 'dashboard_stats'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_date = Column(DateTime, nullable=False, index=True)
    total_attacks = Column(Integer, default=0)
    unique_ips = Column(Integer, default=0)
    top_countries = Column(JSONB, default=dict)
    top_cves = Column(JSONB, default=dict)
    hourly_counts = Column(JSONB, default=dict)
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<DashboardStats(date={self.stat_date}, attacks={self.total_attacks})>"
