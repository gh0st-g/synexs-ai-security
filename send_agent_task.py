#!/usr/bin/env python3
"""
Send Agent Task - Test script for listener.py
Sends tasks to the agent_tasks Redis queue
"""
import redis
import json
import sys

def send_task(task_data: dict):
    """Send a task to the agent queue"""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()  # Test connection

        # Convert task to JSON
        task_json = json.dumps(task_data)

        # Push to queue (RPUSH adds to end, listener uses BLPOP from start)
        r.rpush('agent_tasks', task_json)
        print(f"✅ Task sent to queue: {task_json}")

        # Show queue length
        queue_len = r.llen('agent_tasks')
        print(f"📊 Queue length: {queue_len}")

        return True
    except redis.ConnectionError:
        print("❌ Redis not available on localhost:6379")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
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
