import logging
import pandas as pd
from pathlib import Path
from zenml import step

class IngestData:
    """Handles data ingestion from a CSV file."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path).resolve()  # Ensure absolute path

    def get_data(self) -> pd.DataFrame:
        """Reads data from CSV, handling encoding and errors."""
        try:
            logging.info(f"📥 Reading data from: {self.data_path}")

            df = pd.read_csv(
                self.data_path,
                encoding="latin-1",
                on_bad_lines="skip"  # Skip malformed lines
            )

            if df.empty:
                raise ValueError(f"❌ DataFrame is empty after loading: {self.data_path}")

            logging.info(f"✅ Data successfully loaded. Shape: {df.shape}")
            logging.info(f"🔍 Sample Data:\n{df.sample(min(5, len(df)))}")

            return df

        except FileNotFoundError:
            logging.error(f"❌ File not found: {self.data_path}")
            logging.error("🔹 Ensure the correct path is provided.")
            raise
        except pd.errors.EmptyDataError:
            logging.error(f"❌ File is empty: {self.data_path}")
            raise
        except pd.errors.ParserError:
            logging.error(f"❌ Parsing error in file: {self.data_path}")
            raise
        except Exception as e:
            logging.exception(f"❌ Unexpected error while reading file: {e}")
            raise

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    ZenML step for data ingestion.
    Ensures `data_path` is passed correctly as a string parameter.
    """
    try:
        # Resolve absolute path
        file_path = Path(data_path).resolve()
        logging.info(f"📥 Reading data from: {file_path}")

        # Read CSV
        df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")

        if df.empty:
            raise ValueError(f"❌ DataFrame is empty after loading: {file_path}")

        logging.info(f"✅ Successfully loaded DataFrame: {df.shape}")
        return df

    except FileNotFoundError:
        logging.error(f"❌ File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"❌ File is empty: {file_path}")
        raise
    except Exception as e:
        logging.error(f"❌ Error reading file {file_path}: {e}")
        raise
