#!/usr/bin/env python3
"""
Synexs Database Migration Script
Migrates data from training_binary_v3.jsonl to PostgreSQL
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_db, init_db, test_connection
from db.models import Attack

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_jsonl_to_postgres(jsonl_file="training_binary_v3.jsonl", batch_size=100):
    """
    Migrate existing JSONL training data to PostgreSQL

    Args:
        jsonl_file: Path to JSONL file
        batch_size: Number of records to insert per batch
    """
    if not os.path.exists(jsonl_file):
        logger.warning(f"JSONL file not found: {jsonl_file}")
        return 0

    logger.info(f"Starting migration from {jsonl_file}")

    migrated_count = 0
    batch = []

    with get_db() as db:
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Extract IP from instruction if available
                    ip = data.get('ip', 'unknown')
                    if ip == 'unknown' and 'instruction' in data:
                        # Try to extract IP from instruction: "Host X.X.X.X has vuln..."
                        parts = data['instruction'].split()
                        if len(parts) > 1 and parts[0] == 'Host':
                            ip = parts[1]

                    # Parse binary input
                    binary_input = None
                    if 'input' in data and data['input'].startswith('binary:'):
                        binary_input = data['input'].replace('binary:', '')

                    # Create Attack record
                    attack = Attack(
                        ip=ip,
                        timestamp=datetime.fromtimestamp(data.get('timestamp', datetime.now().timestamp())),
                        vulns=data.get('vulns', []),
                        open_ports=data.get('open_ports', []),
                        actions=data.get('actions', []),
                        binary_input=binary_input,
                        protocol=data.get('protocol', 'v3'),
                        format=data.get('format', 'base64'),
                        instruction=data.get('instruction'),
                        output=data.get('output'),
                        source=data.get('source', 'legacy_migration')
                    )

                    batch.append(attack)

                    # Insert in batches
                    if len(batch) >= batch_size:
                        db.bulk_save_objects(batch)
                        db.commit()
                        migrated_count += len(batch)
                        logger.info(f"Migrated {migrated_count} records...")
                        batch = []

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error migrating line {line_num}: {e}")

        # Insert remaining records
        if batch:
            db.bulk_save_objects(batch)
            db.commit()
            migrated_count += len(batch)

    logger.info(f"Migration complete! Migrated {migrated_count} records")
    return migrated_count


def create_indexes():
    """
    Create additional indexes for performance
    """
    from sqlalchemy import text

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_attacks_country ON attacks(country_code)",
        "CREATE INDEX IF NOT EXISTS idx_attacks_severity ON attacks(severity)",
        "CREATE INDEX IF NOT EXISTS idx_attacks_timestamp_desc ON attacks(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_attacks_vulns_gin ON attacks USING GIN (vulns)",
    ]

    with get_db() as db:
        for idx_sql in indexes:
            try:
                db.execute(text(idx_sql))
                logger.info(f"Created index: {idx_sql[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")


def main():
    """
    Main migration entry point
    """
    logger.info("=" * 60)
    logger.info("Synexs Database Migration")
    logger.info("=" * 60)

    # Test connection
    logger.info("Testing database connection...")
    if not test_connection():
        logger.error("Database connection failed! Check PostgreSQL is running")
        logger.error("Run: docker-compose up -d postgres  OR  systemctl start postgresql")
        sys.exit(1)

    # Initialize database
    logger.info("Initializing database schema...")
    if not init_db():
        logger.error("Failed to initialize database")
        sys.exit(1)

    # Create indexes
    logger.info("Creating performance indexes...")
    create_indexes()

    # Migrate data
    logger.info("Migrating JSONL data to PostgreSQL...")
    count = migrate_jsonl_to_postgres()

    logger.info("=" * 60)
    logger.info(f"Migration complete! {count} records migrated")
    logger.info("Database is ready for use")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
