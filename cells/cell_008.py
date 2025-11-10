import subprocess
import os
import time
import datetime

def ensure_directories():
    print("📁 Ensuring dataset directories exist...")
    required_dirs = [
        "datasets/generated",
        "datasets/to_refine",
        "datasets/refined",
        "datasets/hash_log",
        "datasets/pattern_analysis",
        "datasets/decisions",
        "datasets/mutated",
        "datasets/replicated",
        "datasets/flagged",
        "datasets/discarded",
        "datasets/core_training",
        "datasets/parsed",
        "datasets/routed",
        "datasets/blockchain",
        "memory"
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ All folders are ready.")

def run_cells_once():
    cells = [
        "cell_001.py",
        "cell_002.py",
        "cell_003.py",
        "cell_004.py",
        "cell_005.py",
        "cell_006.py",
        "cell_007.py",
        "cell_009_cleaner.py",
        "cell_010_parser.py",
        "cell_011_router.py",
        # cell_012 runs separately in background
        "cell_013_memory_logger.py",
        "cell_014_mutator.py",
        "cell_015_replicator.py",
        "cell_016_feedback_loop.py"
    ]

    print(f"\n🔁 [cell_008] Starting full Synexs cycle at {datetime.datetime.now().isoformat()}")
    for cell in cells:
        print(f"⚙️ Running {cell}...")
        result = subprocess.run(["python3", cell])
        if result.returncode != 0:
            print(f"❌ Error in {cell}: Exit status {result.returncode}")
        else:
            print(f"✅ {cell} ran successfully.")
    print("⏳ [cell_008] Cycle complete. Waiting 30 seconds...\n")

def main():
    ensure_directories()
    while True:
        run_cells_once()
        time.sleep(30)

if __name__ == "__main__":
    main()
