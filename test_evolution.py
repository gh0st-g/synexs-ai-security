#!/usr/bin/env python3
"""
Test evolution cycle
"""
import sys
import os
import traceback
import time
from ai_swarm_fixed import evolution_cycle, load_cache

# Add current directory to the system path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Change working directory to the app directory
os.chdir(os.path.dirname(__file__))

print("Testing evolution_cycle...")
print(f"Working dir: {os.getcwd()}")

def run_evolution_cycle():
    # Load cache first
    print("\nLoading cache...")
    try:
        load_cache()
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        traceback.print_exc()
        return False

    # Run one cycle
    print("\nRunning evolution_cycle...")
    try:
        evolution_cycle()
        print("✅ Evolution cycle completed!")
        return True
    except Exception as e:
        print(f"❌ Error running evolution_cycle: {e}")
        traceback.print_exc()
        return False

# Run the evolution cycle in a loop with error handling and CPU usage optimization
while True:
    try:
        if run_evolution_cycle():
            # Add a delay to avoid excessive CPU usage
            time.sleep(60)
        else:
            # Add a delay to avoid excessive CPU usage
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        # Add a longer delay to avoid excessive CPU usage
        time.sleep(300)