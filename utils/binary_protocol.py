#!/usr/bin/env python3
"""
Synexs Binary Protocol V3 - Ultra-Compact AI Communication
Achieves 89% size reduction using 5-bit binary encoding
"""

import struct
import json
import base64
from typing import List, Union
from pathlib import Path

# ==================== Action Definitions ====================

# Core actions (5 bits = 32 possible actions)
BINARY_ACTIONS = {
    0x00: "SCAN",         # 00000
    0x01: "ATTACK",       # 00001
    0x02: "REPLICATE",    # 00010
    0x03: "MUTATE",       # 00011
    0x04: "VULN",         # 00100  ← NEW: Required for Shodan/Nmap
    0x05: "LEARN",        # 00101
    0x06: "REPORT",       # 00110
    0x07: "DEFEND",       # 00111
    0x08: "REFINE",       # 01000
    0x09: "FLAG",         # 01001
    0x0A: "XOR_PAYLOAD",  # 01010
    0x0B: "ENCRYPT",      # 01011
    0x0C: "COMPRESS",     # 01100
    0x0D: "HASH_CHECK",   # 01101
    0x0E: "SYNC",         # 01110
    0x0F: "SPLIT",        # 01111
    0x10: "MERGE",        # 10000
    0x11: "STACK_PUSH",   # 10001
    0x12: "STACK_POP",    # 10010
    0x13: "TERMINATE",    # 10011
    0x14: "PAUSE",        # 10100
    0x15: "LOG",          # 10101
    0x16: "QUERY",        # 10110
    0x17: "ACK",          # 10111
    0x18: "NACK",         # 11000
    0x19: "CHECKPOINT",   # 11001
    0x1A: "VALIDATE",     # 11010
    0x1B: "BROADCAST",    # 11011
    0x1C: "LISTEN",       # 11100
    0x1D: "ROUTE",        # 11101
    0x1E: "FILTER",       # 11110
    0x1F: "TRANSFORM",    # 11111
}

# Reverse mapping for encoding
ACTION_TO_BINARY = {v: k for k, v in BINARY_ACTIONS.items()}

# Control tokens
CONTROL_TOKENS = {
    "START": 0x00,
    "END": 0x1F,
    "PAD": 0x00,
}

# ==================== Binary Encoding ====================

def encode_binary(actions: List[str]) -> bytes:
    """
    Encode action list to compact binary format
    Uses 5 bits per action, packed into bytes

    Args:
        actions: List of action names

    Returns:
        Binary-encoded bytes

    Example:
        actions = ["SCAN", "ATTACK", "REPLICATE"]
        binary = encode_binary(actions)
        # Returns: 3 bytes instead of 17 bytes (82% reduction)
    """
    if not actions:
        return b''

    # Convert actions to 5-bit integers
    action_codes = []
    for action in actions:
        code = ACTION_TO_BINARY.get(action)
        if code is None:
            raise ValueError(f"Unknown action: {action}")
        action_codes.append(code)

    # Pack into bytes (5 bits per action)
    bit_string = ''.join(format(code, '05b') for code in action_codes)

    # Pad to byte boundary (8 bits)
    while len(bit_string) % 8 != 0:
        bit_string += '0'

    # Convert bit string to bytes
    binary_data = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

    # Prepend length header (1 byte = max 255 actions)
    num_actions = len(actions)
    return bytes([num_actions]) + binary_data

def decode_binary(binary_data: bytes) -> List[str]:
    """
    Decode binary format back to action list

    Args:
        binary_data: Binary-encoded bytes

    Returns:
        List of action names
    """
    if not binary_data or len(binary_data) < 1:
        return []

    # Extract length header
    num_actions = binary_data[0]
    payload = binary_data[1:]

    # Convert bytes to bit string
    bit_string = ''.join(format(byte, '08b') for byte in payload)

    # Extract 5-bit action codes
    actions = []
    for i in range(num_actions):
        start = i * 5
        end = start + 5
        if end > len(bit_string):
            break
        code = int(bit_string[start:end], 2)
        action = BINARY_ACTIONS.get(code, "UNKNOWN")
        actions.append(action)

    return actions

# ==================== Human-Readable Encoding ====================

def encode_hex(actions: List[str]) -> str:
    """Encode as human-readable hex string"""
    binary = encode_binary(actions)
    return binary.hex()

def decode_hex(hex_string: str) -> List[str]:
    """Decode from hex string"""
    binary = bytes.fromhex(hex_string)
    return decode_binary(binary)

def encode_base64(actions: List[str]) -> str:
    """Encode as base64 string (for JSON/text transport)"""
    binary = encode_binary(actions)
    return base64.b64encode(binary).decode('utf-8')

def decode_base64(b64_string: str) -> List[str]:
    """Decode from base64 string"""
    binary = base64.b64decode(b64_string)
    return decode_binary(binary)

# ==================== Hybrid Mode Support ====================

class HybridProtocol:
    """Support V1 (Greek), V2 (Symbols), V3 (Binary) protocols"""

    def __init__(self, default_protocol="v3"):
        self.protocol = default_protocol

    def encode(self, actions: List[str], protocol: str = None) -> Union[str, bytes]:
        """Encode using specified protocol"""
        proto = protocol or self.protocol

        if proto == "v1":
            mapping = {
                "SCAN": "SIGMA", "ATTACK": "OMEGA", "REPLICATE": "THETA",
                "MUTATE": "DELTA", "EVADE": "ZETA", "LEARN": "ALPHA"
            }
            tokens = [mapping.get(a, "UNKNOWN") for a in actions]
            return " ".join(tokens)

        elif proto == "v2":
            from protocol_v2_proposal import encode_sequence
            return encode_sequence(actions)

        elif proto == "v3":
            return encode_binary(actions)

        elif proto == "v3-hex":
            return encode_hex(actions)

        elif proto == "v3-b64":
            return encode_base64(actions)

        else:
            raise ValueError(f"Unknown protocol: {proto}")

    def decode(self, data: Union[str, bytes], protocol: str = None) -> List[str]:
        """Decode from specified protocol"""
        proto = protocol or self.protocol

        if proto == "v1":
            reverse = {
                "SIGMA": "SCAN", "OMEGA": "ATTACK", "THETA": "REPLICATE",
                "DELTA": "MUTATE", "ZETA": "EVADE", "ALPHA": "LEARN"
            }
            tokens = data.split()
            return [reverse.get(t, "UNKNOWN") for t in tokens]

        elif proto == "v2":
            from protocol_v2_proposal import decode_sequence
            return decode_sequence(data)

        elif proto == "v3":
            return decode_binary(data)

        elif proto == "v3-hex":
            return decode_hex(data)

        elif proto == "v3-b64":
            return decode_base64(data)

        else:
            raise ValueError(f"Unknown protocol: {proto}")

# ==================== Training Data Generator ====================

def generate_binary_training_data(num_samples: int = 1000, format: str = "base64") -> List[dict]:
    """
    Generate training data using binary protocol
    Now includes VULN action
    """
    import random

    training_data = []
    all_actions = list(BINARY_ACTIONS.values())

    for _ in range(num_samples):
        seq_len = random.randint(3, 8)
        actions = random.choices(all_actions, k=seq_len)

        # Encode
        if format == "base64":
            sequence = encode_base64(actions)
        else:
            sequence = encode_binary(actions).hex()

        # Description
        if "VULN" in actions:
            description = f"Host has vulnerability. Exploit chain: {' → '.join(actions[:3])}"
        elif "REPLICATE" in actions and "MUTATE" in actions:
            description = f"Spawn new agent with mutations. Actions: {' → '.join(actions[:3])}"
        elif "SCAN" in actions and "EVADE" in actions:
            description = f"Honeypot detected. Evade immediately. Sequence: {' → '.join(actions[:3])}"
        else:
            description = f"Execute sequence: {' → '.join(actions[:4])}"

        training_data.append({
            "instruction": f"What does binary sequence {sequence[:16]}... mean?",
            "input": f"binary:{sequence}",
            "output": description,
            "actions": actions,
            "protocol": "v3",
            "format": format
        })

    return training_data

# ==================== Export & Main ====================

def export_binary_vocab():
    """Export binary protocol as vocabulary JSON"""
    vocab = {
        "<PAD>": 0x00,
        "<START>": 0x00,
        "<END>": 0x1F,
        "<UNK>": 0x00,
    }
    for code, action in BINARY_ACTIONS.items():
        vocab[action] = code
    return vocab

if __name__ == "__main__":
    print("\nSynexs Binary Protocol V3 - Updated with VULN\n")

    # Test VULN encoding
    test_actions = ["SCAN", "VULN", "EXPLOIT"]
    binary = encode_binary(test_actions)
    b64 = encode_base64(test_actions)
    decoded = decode_binary(binary)

    print(f"Test: {test_actions}")
    print(f"Binary: {binary.hex()} ({len(binary)} bytes)")
    print(f"Base64: {b64}")
    print(f"Decoded: {decoded}")
    print(f"Match: {'PASS' if test_actions == decoded else 'FAIL'}")

    # Generate training data
    samples = generate_binary_training_data(5, format="base64")
    print(f"\nGenerated {len(samples)} samples with VULN support")
    print("Sample:")
    print(json.dumps(samples[0], indent=2))

    # Save
    output_file = Path("training_binary_v3.jsonl")
    full_samples = generate_binary_training_data(1000, format="base64")
    with open(output_file, "w") as f:
        for s in full_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved 1000 samples → {output_file}")

    vocab = export_binary_vocab()
    with open("vocab_v3_binary.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocab saved → vocab_v3_binary.json")
