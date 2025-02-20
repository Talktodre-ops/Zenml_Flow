import logging
from typing import Tuple
import pandas as pd
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing_extensions import Annotated
from zenml import step

@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    ZenML step for data cleaning and preprocessing.
    """
    try:
        logging.info("Starting data cleaning process...")

        # Convert ZenML artifact to DataFrame if needed
        if hasattr(data, "read"):
            data = data.read(pd.DataFrame)
        
        logging.info(f"Initial dataset shape: {data.shape}")

        # Preprocessing strategy
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        logging.info(f"Data after preprocessing: {preprocessed_data.shape}")

        # Splitting strategy
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()

        logging.info(f"Data split completed: X_train={x_train.shape}, X_test={x_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")
        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.exception(f"Error in cleaning data: {e}")
        raise
