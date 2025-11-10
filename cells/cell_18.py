"""
Cell 018 - Agent Generator
This module generates new Synexs agents based on symbolic instruction or templates.
"""

import os
import json
import datetime
import random

# Directory to save generated agents
AGENT_DIR = "agents_generated"
os.makedirs(AGENT_DIR, exist_ok=True)

# Sample agent template (can be expanded)
AGENT_TEMPLATE = {
    "name": "AGENT_NAME",
    "type": "utility",
    "task": "Automate inbox sorting",
    "language": "Python",
    "version": 1.0,
    "created": "CREATED_TIMESTAMP",
    "instructions": [
        "Read inbox messages",
        "Categorize by sender and date",
        "Identify spam",
        "Generate report"
    ],
    "status": "inactive"
}

def generate_agent(name: str, task: str, instructions: list):
    agent = AGENT_TEMPLATE.copy()
    agent["name"] = name
    agent["task"] = task
    agent["instructions"] = instructions
    agent["created"] = datetime.datetime.utcnow().isoformat()

    filename = f"{name.lower().replace(' ', '_')}_{random.randint(1000,9999)}.json"
    path = os.path.join(AGENT_DIR, filename)

    with open(path, "w") as f:
        json.dump(agent, f, indent=4)

    print(f"[cell_018] Agent generated and saved to {path}")
    return path

if __name__ == "__main__":
    print("[cell_018] Agent Generator Initialized.")
    name = input("Agent name: ")
    task = input("Agent task: ")
    print("Enter instructions (type 'done' to finish):")
    instructions = []
    while True:
        line = input("- ")
        if line.lower() == "done":
            break
        instructions.append(line)

    generate_agent(name, task, instructions)
