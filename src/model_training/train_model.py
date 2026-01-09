import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger("src.model_training.train_model")


def load_data() -> pd.DataFrame:
    """Load the feature-engineered training data."""
    train_path = "data/processed/train_processed.csv"
    logger.info(f"Loading feature data from {train_path}")
    train_data = pd.read_csv(train_path)
    return train_data


from typing import Union

def load_params() -> dict[str, Union[float, int]]:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]

def prepare_data(train_data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, OneHotEncoder]:
    """Prepare data for neural network training by separating features and target, and encoding labels."""
    X_train = train_data.drop("target", axis=1).values.astype(np.float32)
    y_train = train_data["target"].values

    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)

    X_tensor = torch.tensor(X_train)
    y_tensor = torch.tensor(y_train_encoded)

    return X_tensor, y_tensor, encoder


class DenseNN(nn.Module):
    """PyTorch equivalent of the Keras Sequential model."""

    def __init__(self, input_shape: int, num_classes: int, params: dict):
        super(DenseNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, params["hidden_layer_1_neurons"])
        self.dropout1 = nn.Dropout(params["dropout_rate"])
        self.fc2 = nn.Linear(params["hidden_layer_1_neurons"], params["hidden_layer_2_neurons"])
        self.dropout2 = nn.Dropout(params["dropout_rate"])
        self.fc3 = nn.Linear(params["hidden_layer_2_neurons"], num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.softmax(self.fc3(x), dim=1)
        return x


def save_training_artifacts(model: nn.Module, encoder: OneHotEncoder) -> None:
    """Save model artifacts to disk."""
    artifacts_dir = "artifacts"
    models_dir = "models"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.pth")
    encoder_path = os.path.join(artifacts_dir, "[target]_one_hot_encoder.joblib")

    logger.info(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)

    logger.info(f"Saving encoder to {encoder_path}")
    joblib.dump(encoder, encoder_path)


def train_model(train_data: pd.DataFrame, params: dict) -> None:
    """Train a PyTorch model, logging metrics and artifacts."""
    torch.manual_seed(params.pop("random_seed"))

    # Prepare data
    X_train, y_train, encoder = prepare_data(train_data)

    dataset = TensorDataset(X_train, y_train)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # Create model
    model = DenseNN(input_shape=X_train.shape[1], num_classes=y_train.shape[1], params=params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    best_val_loss = float("inf")
    patience, patience_counter = 10, 0

    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    logger.info("Training model...")
    for epoch in range(params["epochs"]):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total
        metrics["train_loss"].append(train_loss / len(train_loader))
        metrics["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, torch.argmax(y_batch, dim=1))
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total
        metrics["val_loss"].append(val_loss / len(val_loader))
        metrics["val_acc"].append(val_acc)

        logger.info(f"Epoch {epoch+1}/{params['epochs']} - "
                    f"Train Loss: {metrics['train_loss'][-1]:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {metrics['val_loss'][-1]:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if metrics["val_loss"][-1] < best_val_loss:
            best_val_loss = metrics["val_loss"][-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

    save_training_artifacts(model, encoder)

    # Save metrics
    metrics_path = "metrics/training.json"
    os.makedirs("metrics", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({k: v[-1] for k, v in metrics.items()}, f, indent=2)


def main() -> None:
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()

