import logging
import mlflow
import pandas as pd
import numpy as np
from model.evaluation import MSE, RMSE, R2Score, mean_squared_error
from sklearn.base import ClassifierMixin
from typing import Tuple
from zenml import step
from pydantic import ConfigDict

@step(experiment_tracker="mlflow_tracker")
def evaluation(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:  # ✅ Explicitly define return type as Tuple[float, float]
    """
    Evaluates model performance and logs metrics to MLflow.
    
    Args:
        model: Trained regression model
        x_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple containing MSE and RMSE
    """
    class Config:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    try:
        # Make predictions
        prediction = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, prediction)  # ✅ Ensure float type
        r2_score = float(R2Score().calculate_score(y_test, prediction))
        rmse = np.sqrt(mse)  # ✅ Ensure float type

        # Log metrics to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_metrics({
                "mse": mse,
                "r2_score": r2_score,
                "rmse": rmse
            })

        return mse, rmse  # ✅ Return as float values
        
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        raise e
