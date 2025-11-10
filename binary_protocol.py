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
    0x04: "EVADE",        # 00100
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
    # We'll use a bit array and then convert to bytes
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
            # Greek words
            mapping = {
                "SCAN": "SIGMA", "ATTACK": "OMEGA", "REPLICATE": "THETA",
                "MUTATE": "DELTA", "EVADE": "ZETA", "LEARN": "ALPHA"
            }
            tokens = [mapping.get(a, "UNKNOWN") for a in actions]
            return " ".join(tokens)

        elif proto == "v2":
            # Symbols
            from protocol_v2_proposal import encode_sequence
            return encode_sequence(actions)

        elif proto == "v3":
            # Binary
            return encode_binary(actions)

        elif proto == "v3-hex":
            # Binary as hex (debuggable)
            return encode_hex(actions)

        elif proto == "v3-b64":
            # Binary as base64 (JSON-safe)
            return encode_base64(actions)

        else:
            raise ValueError(f"Unknown protocol: {proto}")

    def decode(self, data: Union[str, bytes], protocol: str = None) -> List[str]:
        """Decode from specified protocol"""
        proto = protocol or self.protocol

        if proto == "v1":
            # Greek words
            reverse = {
                "SIGMA": "SCAN", "OMEGA": "ATTACK", "THETA": "REPLICATE",
                "DELTA": "MUTATE", "ZETA": "EVADE", "ALPHA": "LEARN"
            }
            tokens = data.split()
            return [reverse.get(t, "UNKNOWN") for t in tokens]

        elif proto == "v2":
            # Symbols
            from protocol_v2_proposal import decode_sequence
            return decode_sequence(data)

        elif proto == "v3":
            # Binary
            return decode_binary(data)

        elif proto == "v3-hex":
            # Binary from hex
            return decode_hex(data)

        elif proto == "v3-b64":
            # Binary from base64
            return decode_base64(data)

        else:
            raise ValueError(f"Unknown protocol: {proto}")

# ==================== Training Data Generator ====================

def generate_binary_training_data(num_samples: int = 1000, format: str = "base64") -> List[dict]:
    """
    Generate training data using binary protocol

    Args:
        num_samples: Number of samples to generate
        format: 'binary', 'hex', or 'base64'

    Returns:
        List of training samples
    """
    import random

    training_data = []
    all_actions = list(BINARY_ACTIONS.values())

    for _ in range(num_samples):
        # Random sequence (3-8 actions)
        seq_len = random.randint(3, 8)
        actions = random.choices(all_actions, k=seq_len)

        # Encode based on format
        if format == "binary":
            encoded = encode_binary(actions)
            sequence = encoded.hex()  # Display as hex in JSON
        elif format == "hex":
            sequence = encode_hex(actions)
        elif format == "base64":
            sequence = encode_base64(actions)
        else:
            sequence = encode_binary(actions).hex()

        # Generate description
        if "REPLICATE" in actions and "MUTATE" in actions:
            description = f"Spawn new agent with mutations. Actions: {' → '.join(actions[:3])}"
        elif "SCAN" in actions and "EVADE" in actions:
            description = f"Honeypot detected. Evade immediately. Sequence: {' → '.join(actions[:3])}"
        elif "ATTACK" in actions and "REPORT" in actions:
            description = f"Execute attack and report results. Chain: {' → '.join(actions[:3])}"
        elif "LEARN" in actions:
            description = f"Agent terminated. Learn from failure: {', '.join(actions[:3])}"
        else:
            description = f"Execute sequence: {' → '.join(actions[:4])}"

        training_data.append({
            "instruction": f"What does binary sequence {sequence[:16]}... mean?",
            "input": f"binary:{sequence}",
            "output": description,
            "actions": actions,  # Ground truth for validation
            "protocol": "v3",
            "format": format
        })

    return training_data

# ==================== Performance Comparison ====================

def compare_all_protocols():
    """Compare V1, V2, V3 protocols"""

    # Test message
    actions = ["SCAN", "ATTACK", "REPLICATE", "MUTATE", "EVADE", "LEARN", "REPORT", "DEFEND"]

    hybrid = HybridProtocol()

    # V1: Greek words
    v1_msg = hybrid.encode(actions, "v1")
    v1_size = len(v1_msg.encode('utf-8'))

    # V2: Symbols
    v2_msg = hybrid.encode(actions, "v2")
    v2_size = len(v2_msg.encode('utf-8'))

    # V3: Binary
    v3_msg = hybrid.encode(actions, "v3")
    v3_size = len(v3_msg)

    # V3-hex (debuggable)
    v3_hex = hybrid.encode(actions, "v3-hex")
    v3_hex_size = len(v3_hex.encode('utf-8'))

    # V3-base64 (JSON-safe)
    v3_b64 = hybrid.encode(actions, "v3-b64")
    v3_b64_size = len(v3_b64.encode('utf-8'))

    print("=" * 80)
    print("🚀 PROTOCOL COMPARISON: V1 vs V2 vs V3")
    print("=" * 80)
    print(f"\n📊 Test Message: {len(actions)} actions")
    print(f"Actions: {' → '.join(actions[:4])}...")

    print(f"\n🔴 Protocol V1 (Greek Words):")
    print(f"   Message: {v1_msg}")
    print(f"   Size: {v1_size} bytes")
    print(f"   Efficiency: 1.0x (baseline)")

    print(f"\n🟡 Protocol V2 (Symbols):")
    print(f"   Message: {v2_msg}")
    print(f"   Size: {v2_size} bytes")
    print(f"   Efficiency: {v1_size/v2_size:.2f}x faster")
    print(f"   Reduction: {((v1_size-v2_size)/v1_size)*100:.1f}%")

    print(f"\n🟢 Protocol V3 (Binary):")
    print(f"   Message: {v3_msg.hex()}")
    print(f"   Size: {v3_size} bytes")
    print(f"   Efficiency: {v1_size/v3_size:.2f}x faster")
    print(f"   Reduction: {((v1_size-v3_size)/v1_size)*100:.1f}%")

    print(f"\n🔵 Protocol V3-Hex (Debug):")
    print(f"   Message: {v3_hex}")
    print(f"   Size: {v3_hex_size} bytes")
    print(f"   Efficiency: {v1_size/v3_hex_size:.2f}x faster")
    print(f"   Reduction: {((v1_size-v3_hex_size)/v1_size)*100:.1f}%")

    print(f"\n🟣 Protocol V3-Base64 (JSON-Safe):")
    print(f"   Message: {v3_b64}")
    print(f"   Size: {v3_b64_size} bytes")
    print(f"   Efficiency: {v1_size/v3_b64_size:.2f}x faster")
    print(f"   Reduction: {((v1_size-v3_b64_size)/v1_size)*100:.1f}%")

    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Best compression: V3 Binary ({v3_size} bytes)")
    print(f"Best for JSON: V3-Base64 ({v3_b64_size} bytes)")
    print(f"Best for debug: V3-Hex ({v3_hex_size} bytes)")
    print(f"Most readable: V1 Greek ({v1_size} bytes)")
    print(f"\n💡 Recommendation: Use V3-Base64 for production (JSON-safe + compact)")
    print("=" * 80 + "\n")

    # Verify decode works
    print("🧪 Decode Verification:")
    decoded = hybrid.decode(v3_msg, "v3")
    print(f"   Original: {actions}")
    print(f"   Decoded:  {decoded}")
    print(f"   Match: {'✅ PASS' if actions == decoded else '❌ FAIL'}")
    print()

# ==================== Export Functions ====================

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

# ==================== Main ====================

if __name__ == "__main__":
    print("\n🧠 Synexs Binary Protocol V3 - Ultra-Compact\n")

    # Show comparison
    compare_all_protocols()

    # Generate training samples
    print("📚 Generating Training Data...\n")

    # Base64 format (best for JSON)
    samples_b64 = generate_binary_training_data(10, format="base64")
    print(f"✅ Generated {len(samples_b64)} samples (Base64 format)")
    print("\nSample entries:")
    for i, sample in enumerate(samples_b64[:3], 1):
        print(f"\n{i}. Instruction: {sample['instruction']}")
        print(f"   Input: {sample['input'][:50]}...")
        print(f"   Output: {sample['output']}")
        print(f"   Actions: {len(sample['actions'])} total")

    # Save training data
    output_file = Path("training_binary_v3.jsonl")
    samples_1000 = generate_binary_training_data(1000, format="base64")

    with open(output_file, "w") as f:
        for sample in samples_1000:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved 1000 samples to: {output_file}")

    # Save vocabulary
    vocab_binary = export_binary_vocab()
    vocab_file = Path("vocab_v3_binary.json")
    with open(vocab_file, "w") as f:
        json.dump(vocab_binary, f, indent=2)

    print(f"✅ Saved vocabulary to: {vocab_file}")
    print(f"\n📖 Binary Protocol: {len(BINARY_ACTIONS)} actions (5 bits each)")
    print(f"🎯 Max efficiency: {len(BINARY_ACTIONS)} actions in {len(BINARY_ACTIONS)*5//8 + 1} bytes")
    print(f"💾 Storage savings: Up to 89% vs V1 protocol\n")
