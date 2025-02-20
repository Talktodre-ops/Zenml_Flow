from typing import cast
import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose to run only deployment (`deploy`), only prediction (`predict`), "
         "or both (`deploy_and_predict`). Default is both.",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model.",
)
@click.option(
    "--data-path",
    default=r"C:\Users\damil\IT\customer-satisfaction-mlops-main\data\olist_customers_dataset.csv",
    help="Path to the dataset CSV file.",
)
def main(config: str, min_accuracy: float, data_path: str):
    """Run the MLflow deployment and inference pipeline."""
    # Get the MLflow model deployer component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        # Run the deployment pipeline and pass the dataset path
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
            data_path=data_path,  # Pass the dataset path
        )

    if predict:
        # Run the inference pipeline
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        "You can run:\n"
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'"
        "[/italic green]\n"
        "to inspect experiment runs using the MLflow UI."
    )

    # Check if a model prediction server is already running
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run:\n"
                f"[italic green]zenml model-deployer models delete {service.uuid}[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server failed.\n"
                f"Last state: '{service.status.state.value}'\n"
                f"Error: '{service.status.last_error}'"
            )
    else:
        print(
            "No active MLflow prediction server found. Run with `--config deploy` to train and deploy a model."
        )

if __name__ == "__main__":
    main()
