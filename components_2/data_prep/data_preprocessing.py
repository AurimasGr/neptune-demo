# %%
# import os
# from pathlib import Path

# import matplotlib.pyplot as plt
# import neptune

# from mldesigner import Input, Output, command_component
# from neptune.integrations.xgboost import NeptuneCallback
from utils import *
import argparse


# @command_component(
#     name="prepare_data",
#     display_name="Prepare data",
#     description="Prepare data for training",
#     version="0.1",
#     environment=dict(
#         conda=Path(__file__).parent / "conda.yaml",
#         image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
#     ),
# )
def data_preprocessing_component(
    dataset_path: str,
    train_data: str,
    valid_data: str,
    # dataset: Input(type="uri_folder", description="Dataset"),
    # train_data: Output(type="uri_folder", description="Train data"),
    # valid_data: Output(type="uri_folder", description="Validation data"),
):

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # (neptune) Initialize Neptune run
    # run = neptune.init_run(
    #     tags=["baseline", "xgboost", "walmart-sales"],
    #     name="XGBoost",
    # )

    # Load dataset
    df = pd.read_csv(args.data)
    # load_data(args.data, cache=True)

    # Normalize sales data
    df_normalized = normalize_data(df, "Weekly_Sales")

    # Encode categorical data
    df_encoded = df_normalized.copy()
    df_encoded = encode_categorical_data(df_encoded)

    # Create Lagged features
    df_encoded = create_lags(df_encoded)

    # Split data into train and validation
    X_train, X_valid, y_train, y_valid = get_train_data(df_encoded)

    # Concatenate x and y train data
    train_df = pd.concat([X_train, y_train], axis=1)

    # Concatenate x and y validation data
    valid_df = pd.concat([X_valid, y_valid], axis=1)

    # Save train and validation data
    TRAIN_DATA_PATH = os.path.join(args.train_data, "train_data.csv")
    VALID_DATA_PATH = os.path.join(args.valid_data, "validation_data.csv")

    train_df.to_csv(os.path.join(TRAIN_DATA_PATH, "train_data.csv"), index=False)

    valid_df.to_csv(os.path.join(VALID_DATA_PATH, "validation_data.csv"), index=False)

    # train_df.to_csv(TRAIN_DATA_PATH, index=False)
    # valid_df.to_csv(VALID_DATA_PATH, index=False)


if __name__ == "__main__":
    data_preprocessing_component(
        dataset_path="../data",
        train_data="../data",
        valid_data="../data",
    )