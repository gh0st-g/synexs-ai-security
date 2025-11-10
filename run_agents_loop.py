#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from typing import List

AGENT_DIR = "/app/datasets/agents"
SLEEP_INTERVAL = 15
MAX_AGENTS_PER_BATCH = 10
LOG_FILE = "run_agents_loop.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_agent_files() -> List[str]:
    try:
        agent_files = [os.path.join(AGENT_DIR, f) for f in os.listdir(AGENT_DIR) if f.startswith("sx") and f.endswith(".py")]
        return agent_files
    except Exception as e:
        logging.error(f"Error getting agent files: {e}")
        return []

def run_agents(agent_files: List[str]) -> None:
    for agent in agent_files:
        try:
            subprocess.Popen(
                ["python3", agent],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=os.setpgrp
            )
            logging.info(f"Started agent: {agent}")
        except Exception as e:
            logging.error(f"Error running {agent}: {e}")

def main() -> None:
    while True:
        try:
            agent_files = get_agent_files()
            if not agent_files:
                logging.info("No agents found")
                time.sleep(SLEEP_INTERVAL)
                continue

            selected_agents = agent_files[:min(MAX_AGENTS_PER_BATCH, len(agent_files))]
            run_agents(selected_agents)
            time.sleep(SLEEP_INTERVAL)
        except KeyboardInterrupt:
            logging.info("Exiting due to keyboard interrupt")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(SLEEP_INTERVAL)
            continue

if __name__ == "__main__":
    main()