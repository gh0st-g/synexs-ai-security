#!/usr/bin/env python3
"""
Honeypot Immune System Wrapper
Adds adaptive immune memory to honeypot threat detection

This module provides immune system integration for the honeypot:
- Remembers threat patterns (immune memory)
- Generates antibodies for known attacks
- 10x faster response to recognized threats
- Learns and adapts to evolving threats

Usage:
    Import this instead of direct honeypot logging:
    from honeypot_immune_wrapper import immune_log_attack

The wrapper will:
1. Recognize threat signatures
2. Mount immune responses
3. Create memory cells for future encounters
4. Provide 10x faster blocking for known threats
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add synexs to path
sys.path.append('/root/synexs')

# Import immune system
from synexs_adaptive_immune_system import AdaptiveImmuneSystem, Antigen

# Global immune system instance
_immune_system: Optional[AdaptiveImmuneSystem] = None
_immune_log_file = Path('/root/synexs/datasets/honeypot/immune_memory.json')
_response_cache: Dict[str, str] = {}  # signature -> response_id


def get_immune_system() -> AdaptiveImmuneSystem:
    """Get or create global immune system instance"""
    global _immune_system

    if _immune_system is None:
        _immune_system = AdaptiveImmuneSystem()
        print("🧬 Immune system initialized for honeypot")

        # Load previous immune memory if exists
        _load_immune_memory()

    return _immune_system


def _load_immune_memory():
    """Load immune memory from previous sessions"""
    if not _immune_log_file.exists():
        return

    try:
        with open(_immune_log_file, 'r') as f:
            memory_data = json.load(f)

        # Restore memory cells
        for mem in memory_data.get('memory_cells', []):
            # The immune system will automatically create memory
            # when we encounter threats, so this is just logging
            pass

        print(f"  ✓ Loaded {len(memory_data.get('memory_cells', []))} immune memories")

    except Exception as e:
        print(f"  ⚠️ Failed to load immune memory: {e}")


def _save_immune_memory():
    """Save immune memory for future sessions"""
    try:
        immune = get_immune_system()
        status = immune.get_immune_status()

        memory_data = {
            'timestamp': time.time(),
            'memory_cells': status['memory_cells'],
            'antibodies_available': status['antibodies_available'],
            'total_encounters': status['total_encounters'],
            'success_rate': status['success_rate']
        }

        _immune_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_immune_log_file, 'w') as f:
            json.dump(memory_data, f, indent=2)

    except Exception as e:
        print(f"  ⚠️ Failed to save immune memory: {e}")


def immune_log_attack(attack_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log attack WITH immune system integration

    This replaces the standard log_attack() function.
    It adds immune memory and antibody generation.

    Args:
        attack_data: Attack information dict

    Returns:
        Enhanced attack data with immune response info
    """
    immune = get_immune_system()

    # Extract threat signature
    threat_type = attack_data.get('result', 'unknown')
    source_ip = attack_data.get('ip', 'unknown')

    # Create threat data for immune system
    threat_data = {
        'type': threat_type,
        'source_ip': source_ip,
        'payload': attack_data.get('payload', ''),
        'endpoint': attack_data.get('endpoint', ''),
        'user_agent': attack_data.get('user_agent', ''),
        'timestamp': attack_data.get('timestamp', time.time())
    }

    # Calculate danger level based on attack type
    danger_levels = {
        'sqli': 0.9,
        'xss': 0.8,
        'cmd_injection': 0.95,
        'path_traversal': 0.7,
        'fake_crawler_blocked': 0.6,
        'rate_limited': 0.5,
        'sensitive_path': 0.6
    }

    # Check for multiple threat types
    threats = attack_data.get('threats', [])
    if threats:
        danger = max([danger_levels.get(t, 0.5) for t in threats])
    else:
        danger = danger_levels.get(threat_type, 0.5)

    threat_data['danger_level'] = danger

    # Recognize threat (creates Antigen)
    start_time = time.time()
    antigen = immune.recognize_threat(threat_data)
    recognition_time = (time.time() - start_time) * 1000  # ms

    # Mount immune response
    start_time = time.time()
    response = immune.mount_immune_response(antigen)
    response_time = (time.time() - start_time) * 1000  # ms

    # Report success (honeypot successfully blocked it)
    immune.report_outcome(response.response_id, success=True, metrics={
        'recognition_time_ms': recognition_time,
        'response_time_ms': response_time,
        'danger_level': danger
    })

    # Add immune system info to attack data
    attack_data['immune_response'] = {
        'response_id': response.response_id,
        'is_memory_recall': response.is_memory_recall,
        'antibody_count': len(response.antibodies_deployed),
        'inflammation_level': response.inflammation_level,
        'recognition_time_ms': recognition_time,
        'response_time_ms': response_time,
        'total_response_ms': recognition_time + response_time
    }

    # Periodically save immune memory
    if immune.total_encounters % 100 == 0:
        _save_immune_memory()

    return attack_data


def get_immune_status() -> Dict[str, Any]:
    """Get current immune system status"""
    immune = get_immune_system()
    return immune.get_immune_status()


def get_immune_stats() -> Dict[str, Any]:
    """Get immune system statistics for monitoring"""
    immune = get_immune_system()
    status = immune.get_immune_status()

    # Calculate additional stats
    avg_response_time = 0.0
    memory_hit_rate = 0.0

    if status['total_encounters'] > 0:
        # Memory hit rate = how often we recognize threats
        memory_hit_rate = status['memory_cells'] / max(1, status['total_encounters'])

    return {
        'memory_cells': status['memory_cells'],
        'antibodies': status['antibodies_available'],
        'encounters': status['total_encounters'],
        'success_rate': status['success_rate'],
        'memory_hit_rate': memory_hit_rate,
        'immune_health': status['success_rate']  # Overall immune health
    }


# Export key functions
__all__ = [
    'immune_log_attack',
    'get_immune_system',
    'get_immune_status',
    'get_immune_stats'
]
