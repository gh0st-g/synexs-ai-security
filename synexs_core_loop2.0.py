import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

logging.basicConfig(filename='synexs_core.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def process_cell(cell_data: Tuple[int, int]) -> int:
    cell_id, value = cell_data
    try:
        result = do_processing(value)
        logging.info(f"Processed cell {cell_id} with value {value}, result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error processing cell {cell_id} with value {value}: {str(e)}")
        return recover_from_error(value)

def do_processing(value: int) -> int:
    return value * 2

def recover_from_error(value: int) -> int:
    return value // 2

def main(cell_data: List[Tuple[int, int]]) -> List[int]:
    start_time = time.time()
    max_workers = max(1, os.cpu_count() * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_cell, cell) for cell in cell_data]
        results = [future.result() for future in as_completed(futures)]
    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    return results

def run_continuously(interval: float = 60.0) -> None:
    while True:
        try:
            cell_data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80)]
            results = main(cell_data)
            logging.info(f"Processing results: {results}")
            time.sleep(interval)
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, exiting...")
            break
        except Exception as e:
            logging.error(f"Unexpected error occurred: {str(e)}")
            time.sleep(interval)

if __name__ == "__main__":
    run_continuously()