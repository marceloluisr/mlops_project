import io
import logging
import os

import joblib
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
import yaml
#from train_model import DenseNN, load_params  # import your PyTorch model class + params

logger = logging.getLogger("app.main")


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



class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all artifacts from the local project folder."""
        logger.info("Loading artifacts from local project folder")

        # Define base paths
        artifacts_dir = "artifacts"
        models_dir = "models"

        # Define paths to the preprocessing artifacts
        features_imputer_path = os.path.join(
            artifacts_dir, "[features]_mean_imputer.joblib"
        )
        features_scaler_path = os.path.join(artifacts_dir, "[features]_scaler.joblib")
        target_encoder_path = os.path.join(
            artifacts_dir, "[target]_one_hot_encoder.joblib"
        )
        # Define path to the model file
        model_path = os.path.join(models_dir, "model.pth")

        # Load preprocessing artifacts
        self.features_imputer = joblib.load(features_imputer_path)
        self.features_scaler = joblib.load(features_scaler_path)
        self.target_encoder = joblib.load(target_encoder_path)

        # Load model hyperparameters
        params = load_params()
        input_shape = len(load_breast_cancer().feature_names)
        num_classes = len(self.target_encoder.categories_[0])

        # Load PyTorch model
        self.model = DenseNN(input_shape=input_shape, num_classes=num_classes, params=params)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        logger.info("Successfully loaded all artifacts")

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the full pipeline."""
        # Apply transformations in sequence
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        # Convert to tensor
        X_tensor = torch.tensor(X_scaled.astype("float32"))

        # Get model predictions
        with torch.no_grad():
            y_pred_proba = self.model(X_tensor).numpy()
        y_pred = self.target_encoder.inverse_transform(y_pred_proba)

        return pd.DataFrame({"Prediction": y_pred.ravel()}, index=features.index)


def create_routes(app: Flask) -> None:
    """Create all routes for the application."""

    @app.route("/")
    def index() -> str:
        """Serve the HTML upload interface."""
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Handle CSV file upload, validate features, and return predictions."""
        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            # Read CSV content
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Validate column names against breast cancer dataset
            expected_features = load_breast_cancer().feature_names
            missing_cols = [
                col for col in expected_features if col not in features.columns
            ]
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                )
            features = features[expected_features]

            # Make predictions
            predictions = app.model_service.predict(features)

            # Format predictions for display
            result = predictions.to_string()

            return render_template("index.html", predictions=result)

        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",
            )


# Create and configure Flask app at module level
app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info("Application initialized with model service and routes")


def main() -> None:
    """Run the Flask development server."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()

