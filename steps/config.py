from zenml.config.base_settings import BaseSettings


class ModelNameConfig(BaseSettings):
    """Model Configurations"""

    model_name: str = "LinearRegression"
    fine_tuning: bool = False

# Add this to suppress the warning
    model_config = dict(
        protected_namespaces=()
    )