import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import command
from azure.ai.ml import dsl, Output, Input
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

cpu_compute_target = "cpu-cluster"

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

web_path = "https://raw.githubusercontent.com/neptune-ai/examples/main/use-cases/time-series-forecasting/walmart-sales/dataset/aggregate_data.csv"

aggregate_data = Data(
    name="aggregate_data",
    path=web_path,
    type=AssetTypes.URI_FILE,
    description="Dataset for credit card defaults",
    tags={"source_type": "web", "source": "UCI ML Repo"},
    version="1.0.0",
)

aggregate_data = ml_client.data.create_or_update(aggregate_data)

print(
    f"Dataset with name {aggregate_data.name} was registered to workspace, the dataset version is {aggregate_data.version}"
)

custom_env_name = "aml-scikit-learn"
custom_env_version = "0.1.0"
data_prep_src_dir = "components_2/data_prep"
train_src_dir = "components_2/train"

data_prep_component = command(
    name="data_prep",
    display_name="Data preparation for training",
    description="reads a .csv input, prepares it for training",
    inputs={
        "data": Input(type="uri_folder")
    },
    outputs=dict(
        train_data=Output(type="uri_folder", mode="rw_mount")
    ),
    # The source folder of the component
    code=data_prep_src_dir,
    command="""python data_preprocessing.py \
            --data ${{inputs.data}} \
            --train_data ${{outputs.train_data}}
            """,
    environment=f"{custom_env_name}:{custom_env_version}",
)

train_component = command(
    name="train",
    display_name="Model training",
    description="reads a .csv input, splits into training and validation, trains model and outputs validation dataset",
    inputs={
        "train_data": Input(type="uri_folder")
    },
    outputs=dict(
        valid_data=Output(type="uri_folder", mode="rw_mount")
    ),
    # The source folder of the component
    code=train_src_dir,
    command="""python train.py \
            --data ${{inputs.train_data}} \
            --train_data ${{outputs.valid_data}}
            """,
    environment=f"{custom_env_name}:{custom_env_version}",
)


@dsl.pipeline(
    compute=cpu_compute_target,
    description="E2E data_perp-train pipeline",
)
def ml_pipeline(
    pipeline_job_data_input,
):
    # using data_prep_function like a python call with its own inputs
    data_prep_job = data_prep_component(
        data=pipeline_job_data_input
    )

    # using train_func like a python call with its own inputs
    train_job = train_component(
        train_data=data_prep_job.outputs.train_data,  # note: using outputs from previous step
    )

    # a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {
        "pipeline_job_train_data": data_prep_job.outputs.train_data
    }


pipeline = ml_pipeline(
    pipeline_job_data_input=Input(type="uri_file", path=aggregate_data.path)
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    # Project's name
    experiment_name="e2e_registered_component6",
)

ml_client.jobs.stream(pipeline_job.name)
