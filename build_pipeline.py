import os

# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
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

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

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

# web_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

# credit_data = Data(
#     name="creditcard_defaults",
#     path=web_path,
#     type=AssetTypes.URI_FILE,
#     description="Dataset for credit card defaults",
#     tags={"source_type": "web", "source": "UCI ML Repo"},
#     version="1.0.0",
# )

# credit_data = ml_client.data.create_or_update(credit_data)
# print(
#     f"Dataset with name {credit_data.name} was registered to workspace, the dataset version is {credit_data.version}"
# )

custom_env_name = "aml-scikit-learn"
custom_env_version = "0.1.0"
data_prep_src_dir = "./components/data_prep"
train_src_dir = "./components/train"

from azure.ai.ml import command
from azure.ai.ml import Input, Output

# data_prep_component = command(
#     name="data_prep_credit_defaults",
#     display_name="Data preparation for training",
#     description="reads a .xl input, split the input to train and test",
#     inputs={
#         "data": Input(type="uri_folder"),
#         "test_train_ratio": Input(type="number"),
#     },
#     outputs=dict(
#         train_data=Output(type="uri_folder", mode="rw_mount"),
#         test_data=Output(type="uri_folder", mode="rw_mount"),
#     ),
#     # The source folder of the component
#     code=data_prep_src_dir,
#     command="""python data_prep.py \
#             --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
#             --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
#             """,
#     environment=f"{custom_env_name}:{custom_env_version}",
# )

data_prep_component = command(
    name="data_prep_credit_defaults",
    display_name="Data preparation for training",
    description="reads a .xl input, split the input to train and test",
    inputs={
        "data": Input(type="uri_folder"),
        # "test_train_ratio": Input(type="number"),
    },
    outputs=dict(
        train_data=Output(type="uri_folder", mode="rw_mount"),
        test_data=Output(type="uri_folder", mode="rw_mount"),
    ),
    # The source folder of the component
    code=data_prep_src_dir,
    command="""ls;
               python data_preprocessing.py;
            """,
    environment=f"{custom_env_name}:{custom_env_version}",
)

# importing the Component Package
from azure.ai.ml import load_component

# Loading the component from the yml file
train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))

train_component = ml_client.create_or_update(train_component)

# Create (register) the component in your workspace
print(
    f"Component {train_component.name} with Version {train_component.version} is registered"
)

# the dsl decorator tells the sdk that we are defining an Azure ML pipeline
from azure.ai.ml import dsl, Input, Output

@dsl.pipeline(
    compute=cpu_compute_target,
    description="E2E data_perp-train pipeline",
)
def credit_defaults_pipeline(
    pipeline_job_data_input,
    # pipeline_job_test_train_ratio,
    pipeline_job_learning_rate,
    pipeline_job_registered_model_name,
):
    # using data_prep_function like a python call with its own inputs
    data_prep_job = data_prep_component(
        data=pipeline_job_data_input,
        # test_train_ratio=pipeline_job_test_train_ratio,
    )

    # using train_func like a python call with its own inputs
    train_job = train_component(
        train_data=data_prep_job.outputs.train_data,  # note: using outputs from previous step
        test_data=data_prep_job.outputs.test_data,  # note: using outputs from previous step
        learning_rate=pipeline_job_learning_rate,  # note: using a pipeline input as parameter
        registered_model_name=pipeline_job_registered_model_name,
    )

    # a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {
        "pipeline_job_train_data": data_prep_job.outputs.train_data,
        "pipeline_job_test_data": data_prep_job.outputs.test_data,
    }

registered_model_name = "credit_defaults_model"

# Let's instantiate the pipeline with the parameters of our choice
pipeline = credit_defaults_pipeline(
    # pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
    pipeline_job_data_input=Input(type="uri_file", path=aggregate_data.path),
    # pipeline_job_test_train_ratio=0.24,
    pipeline_job_learning_rate=0.05,
    pipeline_job_registered_model_name=registered_model_name,
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    # Project's name
    experiment_name="e2e_registered_component6",
)

ml_client.jobs.stream(pipeline_job.name)