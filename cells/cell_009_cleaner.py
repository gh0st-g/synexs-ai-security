import os
import time

# Target folders to clean
TARGET_FOLDERS = [
    "datasets/generated",
    "datasets/refined",
    "datasets/decisions",
    "datasets/discarded",
    "datasets/to_refine",
    "datasets/replicated",
    "datasets/mutated",
    "datasets/flagged"
]

# File limit per folder
FILE_LIMIT = 500

def clean_folder(folder):
    if not os.path.exists(folder):
        print(f"[CLEANER] Skipping missing folder: {folder}")
        return

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    if len(files) <= FILE_LIMIT:
        print(f"[CLEANER] {folder} has {len(files)} files. ✅ OK.")
        return

    # Sort by last modified time (oldest first)
    files.sort(key=lambda x: os.path.getmtime(x))

    # Delete excess files
    to_delete = files[:len(files) - FILE_LIMIT]
    for f in to_delete:
        try:
            os.remove(f)
        except Exception as e:
            print(f"[CLEANER] Failed to delete {f}: {e}")

    print(f"[CLEANER] {folder}: Removed {len(to_delete)} old files.")

def main():
    print("\n🧹 [cell_009] Starting cleaner...")
    for folder in TARGET_FOLDERS:
        clean_folder(folder)
    print("✅ [cell_009] Cleanup complete.\n")

if __name__ == "__main__":
    main()
