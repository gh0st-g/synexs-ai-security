#!/usr/bin/env python3
"""
Send Agent Task - Test script for listener.py
Sends tasks to the agent_tasks Redis queue
"""
import os
import redis
import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def send_task(task_data: dict):
    """Send a task to the agent queue"""
    try:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()  # Test connection

        # Convert task to JSON
        task_json = json.dumps(task_data)

        # Push to queue (RPUSH adds to end, listener uses BLPOP from start)
        r.rpush('agent_tasks', task_json)
        logging.info(f"✅ Task sent to queue: {task_json}")

        # Show queue length
        queue_len = r.llen('agent_tasks')
        logging.info(f"📊 Queue length: {queue_len}")

        return True
    except redis.ConnectionError:
        logging.error("❌ Redis not available")
        return False
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    # Example tasks
    if len(sys.argv) > 1:
        # Custom task from command line
        task = {'type': 'custom', 'data': ' '.join(sys.argv[1:])}
    else:
        # Example task
        task = {
            'type': 'image_generation',
            'agent_id': 'agent_001',
            'prompt': 'Generate a landscape image',
            'priority': 'normal'
        }

    send_task(task)