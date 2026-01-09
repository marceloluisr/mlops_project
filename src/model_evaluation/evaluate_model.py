import logging
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
#from train_model import DenseNN, load_params  # import your PyTorch model class
import yaml
logger = logging.getLogger("src.model_evaluation.evaluate_model")


from typing import Union

def load_params() -> dict[str, Union[float, int]]:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]

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




def load_model(input_shape: int, num_classes: int, params: dict) -> torch.nn.Module:
    """Load the trained PyTorch model from disk."""
    model_path = "models/model.pth"
    model = DenseNN(input_shape=input_shape, num_classes=num_classes, params=params)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_encoder():
    """Load the one-hot encoder from disk."""
    encoder_path = "artifacts/[target]_one_hot_encoder.joblib"
    encoder = joblib.load(encoder_path)
    return encoder


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the test dataset from disk."""
    data_path = "data/processed/test_processed.csv"
    logger.info(f"Loading test data from {data_path}")
    data = pd.read_csv(data_path)
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def evaluate_model(
    model: torch.nn.Module, encoder, X: pd.DataFrame, y_true: pd.Series
) -> None:
    """Evaluate the PyTorch model and generate performance metrics."""
    X_tensor = torch.tensor(X.values.astype(np.float32))

    # Forward pass
    with torch.no_grad():
        y_pred_proba = model(X_tensor).numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate evaluation metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    evaluation = {"classification_report": report, "confusion_matrix": cm}

    # Log metrics
    logger.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    evaluation_path = "metrics/evaluation.json"
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    """Main function to orchestrate the model evaluation process."""
    X, y = load_test_data()
    params = load_params()
    encoder = load_encoder()
    model = load_model(input_shape=X.shape[1], num_classes=len(encoder.categories_[0]), params=params)
    evaluate_model(model, encoder, X, y)
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()

