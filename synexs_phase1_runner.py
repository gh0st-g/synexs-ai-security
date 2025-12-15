#!/usr/bin/env python3
import argparse
import signal
import sys
import json
import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from synexs_team_simulator import AgentTeam, MissionStatus
from synexs_training_logger import TrainingDataLogger

# Global flag for graceful shutdown
shutdown_requested = False
logger_instance = None

def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger('synexs_training')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized: {log_file}")
    return logger

def signal_handler(signum, frame):
    global shutdown_requested
    signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
    print(f"\n\n⚠️  {signal_name} received. Finishing current mission and saving progress...")
    if logger_instance:
        logger_instance.warning(f"{signal_name} received - graceful shutdown initiated")
    shutdown_requested = True

class MissionGenerator:
    def __init__(self):
        self.mission_count = 0
        self.environment_templates = {
            'easy': {'risk_level': (0.1, 0.3), 'success_probability': (0.8, 0.95), 'detection_likelihood': (0.05, 0.2), 'honeypot_signals': 0, 'defenses': ['basic_firewall']},
            'medium': {'risk_level': (0.3, 0.6), 'success_probability': (0.5, 0.75), 'detection_likelihood': (0.3, 0.5), 'honeypot_signals': (0, 2), 'defenses': ['firewall', 'IDS']},
            'hard': {'risk_level': (0.6, 0.85), 'success_probability': (0.2, 0.5), 'detection_likelihood': (0.5, 0.8), 'honeypot_signals': (1, 3), 'defenses': ['firewall', 'IDS', 'IPS', 'WAF']},
            'honeypot': {'risk_level': (0.85, 0.95), 'success_probability': (0.05, 0.25), 'detection_likelihood': (0.8, 0.95), 'honeypot_signals': (3, 5), 'defenses': ['firewall', 'IDS', 'IPS', 'honeypot', 'deception_tech']}
        }
        self.signal_types = ['suspicious_timing', 'fake_vulnerabilities', 'too_easy_access', 'honeyd_signature', 'unrealistic_services', 'bait_files', 'fake_credentials']

    def generate(self, difficulty: str = None) -> Dict[str, Any]:
        if difficulty is None:
            difficulty = np.random.choice(['easy', 'medium', 'hard', 'honeypot'], p=[0.2, 0.5, 0.2, 0.1])

        template = self.environment_templates[difficulty]

        risk_level = np.random.uniform(*template['risk_level']) if isinstance(template['risk_level'], tuple) else template['risk_level']
        success_prob = np.random.uniform(*template['success_probability']) if isinstance(template['success_probability'], tuple) else template['success_probability']
        detection = np.random.uniform(*template['detection_likelihood']) if isinstance(template['detection_likelihood'], tuple) else template['detection_likelihood']

        num_signals = np.random.randint(*template['honeypot_signals']) if isinstance(template['honeypot_signals'], tuple) else template['honeypot_signals']
        honeypot_signals = np.random.choice(self.signal_types, size=min(num_signals, len(self.signal_types)), replace=False).tolist()

        environment = {
            'mission_number': self.mission_count,
            'type': f'{difficulty}_network',
            'difficulty': difficulty,
            'topology': {'hosts': np.random.randint(3, 20), 'subnets': np.random.randint(1, 5)},
            'risk_level': risk_level,
            'success_probability': success_prob,
            'detection_likelihood': detection,
            'honeypot_signals': honeypot_signals,
            'defenses': template['defenses'].copy()
        }

        self.mission_count += 1
        return environment

def save_checkpoint(output_dir: str, mission_num: int, stats: Dict[str, Any], team_stats: Dict[str, Any], logger_stats: Dict[str, Any]):
    checkpoint_file = Path(output_dir) / "checkpoint.json"
    checkpoint_temp = Path(output_dir) / "checkpoint.json.tmp"

    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'missions_completed': mission_num,
        'stats': stats,
        'team_stats': team_stats,
        'logger_stats': logger_stats
    }

    try:
        with open(checkpoint_temp, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        checkpoint_temp.replace(checkpoint_file)
    except Exception as e:
        if logger_instance:
            logger_instance.error(f"Failed to save checkpoint: {e}")
        raise

def save_progress_file(output_dir: str, mission_num: int, total_missions: int, stats: Dict[str, Any], start_time: float):
    progress_file = Path(output_dir) / "progress.json"
    progress_temp = Path(output_dir) / "progress.json.tmp"

    elapsed = time.time() - start_time
    rate = (mission_num / elapsed) if elapsed > 0 else 0
    remaining = ((total_missions - mission_num) / rate) if rate > 0 else 0

    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'mission_current': mission_num,
        'mission_total': total_missions,
        'progress_percent': (mission_num / total_missions * 100) if total_missions > 0 else 0,
        'elapsed_seconds': elapsed,
        'rate_missions_per_sec': rate,
        'eta_seconds': remaining,
        'stats': stats,
        'status': 'running'
    }

    try:
        with open(progress_temp, 'w') as f:
            json.dump(progress_data, f, indent=2)
        progress_temp.replace(progress_file)
    except Exception as e:
        if logger_instance:
            logger_instance.error(f"Failed to save progress file: {e}")

def calculate_checkpoint_interval(num_missions: int) -> int:
    if num_missions <= 100:
        return 10
    elif num_missions <= 1000:
        return 50
    elif num_missions <= 10000:
        return 100
    else:
        return 500

def load_checkpoint(output_dir: str) -> Optional[Dict[str, Any]]:
    checkpoint_file = Path(output_dir) / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                if logger_instance:
                    logger_instance.info(f"Checkpoint loaded: {data.get('missions_completed', 0)} missions")
                return data
        except json.JSONDecodeError as e:
            msg = f"Checkpoint file corrupted: {e}. Starting fresh."
            print(f"⚠️  Warning: {msg}")
            if logger_instance:
                logger_instance.error(msg)
        except Exception as e:
            msg = f"Could not load checkpoint: {e}"
            print(f"⚠️  Warning: {msg}")
            if logger_instance:
                logger_instance.error(msg)
    return None

def run_training_session(num_missions: int = 100, output_dir: str = "./training_logs", team_id: str = None, resume: bool = True):
    global shutdown_requested, logger_instance

    logger = setup_logging(output_dir)
    logger_instance = logger

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("="*70)
    logger.info("SYNEXS PHASE 1 - TEAM TRAINING SESSION")
    logger.info("="*70)

    print("\n" + "="*70)
    print("SYNEXS PHASE 1 - TEAM TRAINING SESSION")
    print("="*70)

    checkpoint_interval = calculate_checkpoint_interval(num_missions)
    logger.info(f"Checkpoint interval: every {checkpoint_interval} missions")

    start_mission = 0
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            start_mission = checkpoint['missions_completed']
            print(f"\n📂 Checkpoint found! Resuming from mission {start_mission + 1}")
            logger.info(f"Resuming from checkpoint at mission {start_mission + 1}")

    print(f"\nConfiguration:")
    print(f"  Missions to run: {num_missions}")
    print(f"  Starting from: Mission {start_mission + 1}")
    print(f"  Output directory: {output_dir}")
    print(f"  Team ID: {team_id or 'auto-generated'}")
    print(f"  Checkpoint interval: {checkpoint_interval} missions")
    print("="*70 + "\n")

    logger.info(f"Configuration: missions={num_missions}, start={start_mission+1}, output={output_dir}")

    try:
        logger.info("Initializing team, data logger, and mission generator...")
        team = AgentTeam(team_id=team_id)
        data_logger = TrainingDataLogger(output_dir=output_dir)
        generator = MissionGenerator()
        logger.info("All components initialized successfully")
    except Exception as e:
        error_msg = f"Error initializing components: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        sys.exit(1)

    start_time = time.time()
    if checkpoint and 'stats' in checkpoint:
        success_count = checkpoint['stats'].get('success_count', 0)
        failure_count = checkpoint['stats'].get('failure_count', 0)
        abort_count = checkpoint['stats'].get('abort_count', 0)
    else:
        success_count = 0
        failure_count = 0
        abort_count = 0

    generator.mission_count = start_mission
    logger.info(f"Starting mission loop from mission {start_mission + 1}")

    missions_completed = start_mission
    for i in range(start_mission, num_missions):
        if shutdown_requested:
            logger.warning("Shutdown requested - breaking mission loop")
            print("\n\n🛑 Graceful shutdown initiated...")
            break

        print(f"\n{'─'*70}")
        print(f"Mission {i+1}/{num_missions}")
        print(f"{'─'*70}")

        try:
            logger.debug(f"Mission {i+1}: Generating environment")
            environment = generator.generate()
            print(f"Difficulty: {environment['difficulty']}")
            print(f"Risk: {environment['risk_level']:.2f} | Success Prob: {environment['success_probability']:.2f} | Detection: {environment['detection_likelihood']:.2f}")

            logger.debug(f"Mission {i+1}: Assigning to team")
            mission_id = team.assign_mission({'type': environment['type'], 'number': i + 1})

            logger.debug(f"Mission {i+1}: Executing mission {mission_id}")
            result = team.execute_mission(mission_id, environment)

            logger.debug(f"Mission {i+1}: Logging results")
            data_logger.log_mission(result)

            if result.status == MissionStatus.SUCCESS:
                success_count += 1
                logger.debug(f"Mission {i+1}: SUCCESS")
            elif result.status == MissionStatus.FAILURE:
                failure_count += 1
                logger.debug(f"Mission {i+1}: FAILURE")
            else:
                abort_count += 1
                logger.debug(f"Mission {i+1}: ABORTED")

            missions_completed = i + 1

        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_msg = f"Error during mission {i+1}: {e}"
            print(f"❌ {error_msg}")
            print(f"   Continuing with next mission...")
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            failure_count += 1
            missions_completed = i + 1
            continue

        if (i + 1) % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1 - start_mission) / elapsed if elapsed > 0 else 0
            remaining = (num_missions - i - 1) / rate if rate > 0 else 0

            print(f"\n{'='*70}")
            print(f"Progress: {i+1}/{num_missions} missions ({(i+1)/num_missions*100:.1f}%)")
            print(f"  Success: {success_count} | Failure: {failure_count} | Abort: {abort_count}")
            print(f"  Rate: {rate:.2f} missions/sec")
            print(f"  Est. time remaining: {remaining:.1f}s ({remaining/60:.1f}m)")
            print(f"{'='*70}")

            logger.info(f"Progress: {i+1}/{num_missions} ({(i+1)/num_missions*100:.1f}%) - Rate: {rate:.2f} m/s - ETA: {remaining/60:.1f}m")

            try:
                current_stats = {
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'abort_count': abort_count,
                    'elapsed_time': elapsed
                }
                save_checkpoint(output_dir, i + 1, current_stats, team.get_team_stats(), data_logger.get_statistics())
                print(f"💾 Checkpoint saved at mission {i+1}")
                logger.debug(f"Checkpoint saved at mission {i+1}")
            except Exception as e:
                error_msg = f"Could not save checkpoint: {e}"
                print(f"⚠️  Warning: {error_msg}")
                logger.error(error_msg)
                logger.error(traceback.format_exc())

        if (i + 1) % max(10, checkpoint_interval // 5) == 0:
            try:
                current_stats = {
                    'success_count': success_count,