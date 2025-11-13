import logging
import logging.handlers
import time
import signal
import sys
import os
import fcntl
import json
import threading
from datetime import datetime
from typing import Optional
from contextlib import suppress
from http.server import HTTPServer, BaseHTTPRequestHandler

# PID file for singleton enforcement
PID_FILE = '/tmp/listener.pid'
lock_fp = None

# Set up logging with rotation
handler = logging.handlers.RotatingFileHandler(
    'listener.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logging.basicConfig(handlers=[handler], level=logging.INFO)

# Global variables
running = True
message_count = 0
error_count = 0
start_time = datetime.now()
last_message_time = None

# Health endpoint
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            uptime = (datetime.now() - start_time).total_seconds()
            health_data = {
                'status': 'healthy' if running else 'shutting_down',
                'uptime_seconds': uptime,
                'messages_processed': message_count,
                'errors': error_count,
                'last_message': last_message_time.isoformat() if last_message_time else None,
                'pid': os.getpid()
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_data, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs

def start_health_server():
    """Start health check HTTP server on port 8765 in background thread"""
    try:
        server = HTTPServer(('0.0.0.0', 8765), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logging.info('Health endpoint started on http://0.0.0.0:8765/health')
    except Exception as e:
        logging.warning(f'Could not start health endpoint: {e}')

def acquire_pid_lock():
    """Acquire exclusive PID lock to ensure only one instance runs"""
    global lock_fp
    try:
        lock_fp = open(PID_FILE, 'x')
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fp.write(str(os.getpid()))
        lock_fp.flush()
        logging.info(f'Acquired PID lock: {os.getpid()}')
        return True
    except IOError:
        logging.error("Another listener instance is already running!")
        logging.error(f"Check PID file: {PID_FILE}")
        return False

def release_pid_lock():
    """Release PID lock and remove PID file"""
    global lock_fp
    if lock_fp:
        try:
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
            lock_fp.close()
            os.remove(PID_FILE)
            logging.info('Released PID lock')
        except Exception as e:
            logging.warning(f'Error releasing PID lock: {e}')

def signal_handler(sig, frame):
    global running
    logging.info('Received shutdown signal, stopping listener...')
    running = False

try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except (ValueError, RuntimeError):
    pass

def listen():
    global message_count, error_count, last_message_time
    while running:
        try:
            # Listen for messages
            message = receive_message()
            if message:
                process_message(message)
                message_count += 1
                last_message_time = datetime.now()
        except Exception as e:
            error_count += 1
            logging.error(f'Error processing message: {e}', exc_info=True)
            time.sleep(5)  # Avoid busy loop and retry after 5 seconds

    logging.info(f'Listener stopped. Total messages processed: {message_count}')

def receive_message() -> Optional[str]:
    """
    Receive message from Redis queue with blocking pop (efficient, <1% CPU idle)
    Uses BLPOP with 5s timeout - returns None if no message, waits efficiently
    """
    try:
        import redis
        # Lazy connection on first call
        if not hasattr(receive_message, 'redis_client'):
            receive_message.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=2
            )
            receive_message.redis_client.ping()  # Test connection
            logging.info('Connected to Redis on localhost:6379')

        # BLPOP blocks efficiently until message arrives (or 5s timeout)
        # This is the key: uses ~0% CPU when idle, unlike tight loop
        result = receive_message.redis_client.blpop('agent_tasks', timeout=5)

        if result:
            queue_name, message = result
            return message
        return None  # No message in 5 seconds, loop continues efficiently

    except redis.ConnectionError:
        logging.warning('Redis not available, falling back to sleep')
        time.sleep(5)  # Fallback: sleep if Redis is down
        return None
    except Exception as e:
        logging.error(f'Error receiving message: {e}')
        time.sleep(5)
        return None

def process_message(message: str):
    # Implement message processing logic here
    logging.info(f'Processed message: {message}')

def main():
    # Ensure only one instance runs
    if not acquire_pid_lock():
        return

    try:
        logging.info('Starting listener...')
        start_health_server()  # Start health endpoint on port 8765
        listen()
    except (KeyboardInterrupt, SystemExit):
        logging.info('Received shutdown signal, stopping listener...')
    except Exception as e:
        logging.error(f'Unhandled exception: {e}', exc_info=True)
        sys.exit(1)
    finally:
        release_pid_lock()
        logging.info('Listener stopped.')

if __name__ == '__main__':
    main()