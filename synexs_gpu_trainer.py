#!/usr/bin/env python3
"""
Synexs GPU Training Pipeline
Trains neural network on team coordination data using PyTorch + GPU

This module:
- Loads batched training data
- Trains models on GPU (if available)
- Evaluates mission success prediction
- Saves trained models for deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime
import time

from synexs_training_logger import GPUTrainingDataset


class MissionPredictor(nn.Module):
    """
    Neural network that predicts mission outcomes

    Architecture:
    - Input: [batch, sequence_length, feature_dim]
    - LSTM layers for sequence processing
    - Attention mechanism
    - Fully connected layers
    - Output: [batch, num_classes] (SUCCESS/FAILURE/ABORT)
    """

    def __init__(self, feature_dim: int = 19, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 3):
        super(MissionPredictor, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [batch, sequence_length, feature_dim]

        Returns:
            output: [batch, num_classes]
            attention_weights: [batch, sequence_length]
        """
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: [batch, sequence_length, hidden_dim]

        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize

        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_dim]

        # Fully connected layers
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out, attention_weights.squeeze(-1)


class GPUTrainer:
    """Manages training on GPU"""

    def __init__(self, model: nn.Module, device: str = None):
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        print(f"✓ Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs, attention = self.model(features)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs, attention = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, early_stopping_patience: int = 10):
        """Full training loop with early stopping"""
        print(f"\n{'='*60}")
        print(f"Starting training: {num_epochs} epochs")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  ✓ New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Final train accuracy: {self.train_accuracies[-1]:.2f}%")
        print(f"  Final val accuracy: {self.val_accuracies[-1]:.2f}%")
        print(f"{'='*60}\n")

    def save_model(self, save_path: str):
        """Save trained model"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'device': str(self.device),
            'model_architecture': {
                'feature_dim': self.model.feature_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'num_classes': self.model.num_classes
            }
        }, save_path)

        print(f"✓ Model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load trained model"""
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']

        print(f"✓ Model loaded from {load_path}")

    def predict(self, features: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        Make prediction for a single mission

        Returns:
            predicted_class: 0 (SUCCESS), 1 (FAILURE), 2 (ABORTED)
            confidence: probability of predicted class
            attention_weights: what the model focused on
        """
        self.model.eval()

        with torch.no_grad():
            features = features.unsqueeze(0).to(self.device)  # Add batch dimension
            outputs, attention = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence, attention.squeeze(0)


def create_training_pipeline(data_dir: str, output_dir: str = "./models",
                             batch_size: int = 32, num_epochs: int = 50):
    """
    Complete training pipeline

    Args:
        data_dir: Directory containing training batches
        output_dir: Where to save trained models
        batch_size: Batch size for DataLoader
        num_epochs: Number of training epochs
    """
    print("\n" + "="*60)
    print("SYNEXS GPU TRAINING PIPELINE")
    print("="*60 + "\n")

    # Load dataset
    print("Loading dataset...")
    dataset = GPUTrainingDataset(data_dir)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("\nInitializing model...")
    model = MissionPredictor(
        feature_dim=19,  # Match feature extraction in logger
        hidden_dim=128,
        num_layers=2,
        num_classes=3
    )

    # Create trainer
    trainer = GPUTrainer(model)

    # Train
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{output_dir}/synexs_mission_predictor_{timestamp}.pt"
    trainer.save_model(model_path)

    # Save training report
    report = {
        'timestamp': timestamp,
        'model_path': model_path,
        'train_size': train_size,
        'val_size': val_size,
        'final_train_accuracy': trainer.train_accuracies[-1],
        'final_val_accuracy': trainer.val_accuracies[-1],
        'best_val_loss': min(trainer.val_losses),
        'total_epochs': len(trainer.train_losses),
        'device': str(trainer.device)
    }

    report_path = f"{output_dir}/training_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Training report saved to {report_path}")

    return trainer, report


def main():
    """Test GPU training pipeline"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python synexs_gpu_trainer.py <data_dir>")
        print("\nExample:")
        print("  python synexs_gpu_trainer.py ./training_logs/batches")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not Path(data_dir).exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Run training pipeline
    trainer, report = create_training_pipeline(
        data_dir=data_dir,
        output_dir="./models",
        batch_size=16,  # Smaller batch for testing
        num_epochs=30
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nFinal Results:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("\n")


if __name__ == "__main__":
    main()
