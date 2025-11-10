#!/usr/bin/env python3
"""
Improved AI swarm test script
"""
import sys
import os
import logging
from ai_swarm_fixed import send_telegram, test_connections

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('test_swarm.log', mode='a'),
        logging.StreamHandler()
    ]
)

def main():
    try:
        logging.info("Testing AI swarm components...")

        # Test working directory
        current_dir = os.getcwd()
        logging.info("1. Current dir: %s", current_dir)
        if current_dir != "/app":
            try:
                os.chdir("/app")
                logging.info("   Changed to: %s", os.getcwd())
            except OSError as e:
                logging.error("Error changing directory: %s", e)
                return 1

        # Test Telegram
        logging.info("2. Testing Telegram...")
        try:
            result = send_telegram("🧪 <b>Manual Test</b>\nAI Swarm debugging", force=True)
            logging.info("   Result: %s", '✅ SUCCESS' if result else '❌ FAILED')
        except Exception as e:
            logging.error("Error sending Telegram message: %s", e)
            return 1

        # Test connection function
        logging.info("3. Testing connection function...")
        try:
            conn_result = test_connections()
            logging.info("   Result: %s", '✅ PASS' if conn_result else '❌ FAIL')
        except Exception as e:
            logging.error("Error testing connections: %s", e)
            return 1

        logging.info("✅ Component tests complete")
        return 0
    except Exception as e:
        logging.critical("Unhandled exception: %s", e, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())