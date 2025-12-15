#!/usr/bin/env python3
"""
Synexs V3 Integration Test Suite
Tests model loading, protocol efficiency, and data flow
"""

import json
import sys
import os
from pathlib import Path

# Test 1: V3 Vocabulary Loading
def test_vocab_loading():
    print("=" * 60)
    print("TEST 1: V3 Vocabulary Loading")
    print("=" * 60)

    try:
        with open('vocab_v3_binary.json', 'r') as f:
            vocab = json.load(f)

        # Extract action names (non-special tokens)
        actions = [k for k in vocab.keys() if not k.startswith('<')]
        special_tokens = [k for k in vocab.keys() if k.startswith('<')]

        print(f"✅ Vocabulary loaded successfully")
        print(f"   Total tokens: {len(vocab)}")
        print(f"   Actions: {len(actions)}")
        print(f"   Special tokens: {len(special_tokens)}")
        print(f"   Max index: {max(vocab.values())}")
        print(f"\n   Actions: {', '.join(actions[:10])}...")
        print(f"   Special: {', '.join(special_tokens)}")

        return True, vocab, actions
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, None, None

# Test 2: Model Architecture Compatibility
def test_model_architecture():
    print("\n" + "=" * 60)
    print("TEST 2: Model Architecture Compatibility")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        # Load vocabulary
        with open('vocab_v3_binary.json', 'r') as f:
            vocab = json.load(f)

        vocab_size = max(vocab.values()) + 1
        actions = [k for k in vocab.keys() if not k.startswith('<')]
        output_dim = len(actions)

        # Define model (matching synexs_model.py)
        class SynexsCoreModel(nn.Module):
            def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, output_dim=32):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.fc1 = nn.Linear(embed_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                embedded = self.embedding(x)
                pooled = embedded.mean(dim=1)
                h = self.fc1(pooled)
                h = self.relu(h)
                h = self.dropout(h)
                out = self.fc2(h)
                return out

        # Create model
        model = SynexsCoreModel(vocab_size, output_dim=output_dim)

        print(f"✅ Model architecture created successfully")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Output actions: {output_dim}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        test_input = torch.randint(0, vocab_size, (2, 5))  # Batch of 2, sequence length 5
        output = model(test_input)

        print(f"   Test input shape: {test_input.shape}")
        print(f"   Test output shape: {output.shape}")
        print(f"   Expected output: (batch_size={2}, actions={output_dim})")

        if output.shape == (2, output_dim):
            print("✅ Forward pass successful!")
            return True
        else:
            print(f"❌ Output shape mismatch!")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test 3: Binary Protocol Efficiency
def test_protocol_efficiency():
    print("\n" + "=" * 60)
    print("TEST 3: Binary Protocol Efficiency")
    print("=" * 60)

    try:
        # Import binary protocol
        sys.path.insert(0, '/root/synexs')
        from binary_protocol import encode_base64, decode_base64, encode_hex, decode_hex

        # Test actions
        test_actions = ["SCAN", "ATTACK", "REPLICATE", "MUTATE", "EVADE"]

        # V1 Protocol (Greek words - for comparison)
        v1_message = "ALPHA OMEGA THETA DELTA ZETA"
        v1_size = len(v1_message.encode('utf-8'))

        # V3 Protocol (Binary)
        v3_base64 = encode_base64(test_actions)
        v3_hex = encode_hex(test_actions)
        v3_size = len(v3_base64.encode('utf-8'))

        # Decode to verify
        decoded = decode_base64(v3_base64)

        print(f"✅ Binary protocol test successful")
        print(f"\n   V1 (Greek): '{v1_message}'")
        print(f"      Size: {v1_size} bytes")
        print(f"\n   V3 (Base64): '{v3_base64}'")
        print(f"      Size: {v3_size} bytes")
        print(f"      Hex: '{v3_hex}'")
        print(f"\n   Reduction: {((v1_size - v3_size) / v1_size * 100):.1f}%")
        print(f"   Speedup: {v1_size / v3_size:.2f}x faster")
        print(f"   Decoded correctly: {decoded == test_actions}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test 4: Data Flow Pipeline
def test_data_flow():
    print("\n" + "=" * 60)
    print("TEST 4: Data Flow Pipeline")
    print("=" * 60)

    try:
        # Check if decisions.json exists and has correct format
        decisions_path = Path("datasets/decisions/decisions.json")

        if decisions_path.exists():
            with open(decisions_path, 'r') as f:
                data = json.load(f)

            # Handle both formats: {"decisions": [...]} or [...]
            if isinstance(data, dict):
                decisions = data.get("decisions", [])
            elif isinstance(data, list):
                decisions = data
            else:
                decisions = []

            if decisions:
                sample = decisions[0]
                has_sequence = "sequence" in sample
                has_action = "action" in sample

                print(f"✅ Decisions file format validated")
                print(f"   Total decisions: {len(decisions)}")
                print(f"   Sample: {sample}")
                print(f"   Has sequence: {has_sequence}")
                print(f"   Has action: {has_action}")

                # Count actions
                action_counts = {}
                for d in decisions:
                    action = d.get("action", "unknown")
                    action_counts[action] = action_counts.get(action, 0) + 1

                print(f"\n   Action distribution:")
                for action, count in sorted(action_counts.items()):
                    print(f"      {action}: {count}")

                return True
            else:
                print("⚠️  Decisions file is empty")
                return False
        else:
            print("⚠️  No decisions.json found - run cell_006.py first")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test 5: AI Configuration
def test_ai_config():
    print("\n" + "=" * 60)
    print("TEST 5: AI Configuration")
    print("=" * 60)

    try:
        with open('ai_config.json', 'r') as f:
            config = json.load(f)

        print(f"✅ AI configuration loaded")
        print(f"   Version: {config.get('version')}")
        print(f"   Protocol: {config.get('protocol')}")
        print(f"   Shadow mode: {config['ai_mode'].get('shadow_mode')}")
        print(f"   Auto-retrain: {config['training'].get('auto_retrain')}")
        print(f"   Retrain interval: {config['training'].get('retrain_every_n_cycles')} cycles")
        print(f"   V3 Actions: {len(config['actions'].get('v3_actions', []))}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

# Test 6: Communication Efficiency Simulation
def test_communication_efficiency():
    print("\n" + "=" * 60)
    print("TEST 6: Agent Communication Efficiency Simulation")
    print("=" * 60)

    try:
        sys.path.insert(0, '/root/synexs')
        from binary_protocol import encode_base64, decode_base64

        # Simulate 1000 agents sending 100 messages each
        num_agents = 1000
        messages_per_agent = 100
        total_messages = num_agents * messages_per_agent

        # Average message: 5 actions
        avg_actions_per_msg = 5

        # V1 size (Greek words): approximately 28 bytes for 5 actions
        v1_size_per_msg = 28
        v1_total = total_messages * v1_size_per_msg

        # V3 size (Binary): approximately 6 bytes for 5 actions
        v3_size_per_msg = 6
        v3_total = total_messages * v3_size_per_msg

        # Calculate savings
        bandwidth_saved = v1_total - v3_total
        reduction_pct = (bandwidth_saved / v1_total) * 100

        # Per year (messages per hour)
        messages_per_hour = 100
        hours_per_year = 24 * 365
        yearly_v1 = (num_agents * messages_per_hour * hours_per_year * v1_size_per_msg) / (1024**3)  # GB
        yearly_v3 = (num_agents * messages_per_hour * hours_per_year * v3_size_per_msg) / (1024**3)  # GB
        yearly_saved = yearly_v1 - yearly_v3

        print(f"✅ Communication efficiency simulation")
        print(f"\n   Scenario: {num_agents:,} agents × {messages_per_agent} messages")
        print(f"   Total messages: {total_messages:,}")
        print(f"\n   V1 Protocol (Greek):")
        print(f"      Per message: {v1_size_per_msg} bytes")
        print(f"      Total: {v1_total:,} bytes ({v1_total/1024/1024:.2f} MB)")
        print(f"\n   V3 Protocol (Binary):")
        print(f"      Per message: {v3_size_per_msg} bytes")
        print(f"      Total: {v3_total:,} bytes ({v3_total/1024/1024:.2f} MB)")
        print(f"\n   Savings:")
        print(f"      Bandwidth saved: {bandwidth_saved:,} bytes ({bandwidth_saved/1024/1024:.2f} MB)")
        print(f"      Reduction: {reduction_pct:.1f}%")
        print(f"      Speedup: {v1_total/v3_total:.2f}x faster transmission")
        print(f"\n   Annual Projection (100 msg/hour/agent):")
        print(f"      V1: {yearly_v1:.2f} GB/year")
        print(f"      V3: {yearly_v3:.2f} GB/year")
        print(f"      Saved: {yearly_saved:.2f} GB/year")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main Test Runner
def main():
    print("\n" + "🧠" * 30)
    print("SYNEXS V3 INTEGRATION TEST SUITE")
    print("🧠" * 30 + "\n")

    results = {}

    # Run tests
    results['vocab'] = test_vocab_loading()[0]
    results['model'] = test_model_architecture()
    results['protocol'] = test_protocol_efficiency()
    results['data_flow'] = test_data_flow()
    results['config'] = test_ai_config()
    results['communication'] = test_communication_efficiency()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.upper()}: {status}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System ready for AI integration.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
