import logging
import logging.handlers
import time
import signal
import sys
import os
import fcntl
from typing import Optional
from contextlib import suppress

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
    global message_count
    while running:
        try:
            # Listen for messages
            message = receive_message()
            if message:
                process_message(message)
                message_count += 1
        except Exception as e:
            logging.error(f'Error processing message: {e}', exc_info=True)
            time.sleep(5)  # Avoid busy loop and retry after 5 seconds

    logging.info(f'Listener stopped. Total messages processed: {message_count}')

def receive_message() -> Optional[str]:
    # Implement message receiving logic here
    return 'sample_message'

def process_message(message: str):
    # Implement message processing logic here
    logging.info(f'Processed message: {message}')

def main():
    # Ensure only one instance runs
    if not acquire_pid_lock():
        return

    try:
        logging.info('Starting listener...')
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