import os

from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

CUSTOM_ENV_NAME = "neptune-example"
DEPENDENCIES_DIR = "./dependencies"


def main():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id="e1685108-7640-4417-8331-c2a3f43bc109",
        resource_group_name="neptune",
        workspace_name="aurimas-test-1",
    )

    pipeline_job_env = Environment(
        name=CUSTOM_ENV_NAME,
        description="Custom environment for Neptune Example",
        tags={"scikit-learn": "0.24.2"},
        conda_file=os.path.join(DEPENDENCIES_DIR, "conda.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        version="0.2.0",
    )
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    print(
        f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
    )


if __name__ == "__main__":
    main()
