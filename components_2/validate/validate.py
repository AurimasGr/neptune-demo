import argparse
import os

import joblib
import neptune
import pandas as pd
import seaborn as sns
import xgboost as xgb
from azureml.core import Run
from matplotlib import pyplot as plt
from neptune.types import File
from sklearn.metrics import mean_absolute_error, mean_squared_error


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_data", type=str, help="path to validation data")
    parser.add_argument("--neptune_project", type=str, help="neptune project to log to")
    parser.add_argument("--neptune_custom_run_id", type=str, help="neptune run id to log to")
    parser.add_argument("--neptune_api_token", type=str, help="neptune service account token")
    args = parser.parse_args()

    os.environ["NEPTUNE_PROJECT"] = args.neptune_project
    os.environ["NEPTUNE_CUSTOM_RUN_ID"] = args.neptune_custom_run_id
    os.environ["NEPTUNE_API_TOKEN"] = args.neptune_api_token
    valid_df = pd.read_csv(select_first_file(args.valid_data))

    # Get train data
    X_valid, y_valid = valid_df.drop(["Weekly_Sales"], axis=1), valid_df.Weekly_Sales

    # (neptune) Initialize Neptune run
    run = neptune.init_run(
        tags=["MLOps", "baseline", "xgboost", "walmart-sales"],
        name="XGBoost",
    )

    # Load model checkpoint from model registry
    run["training/model"].download()
    model_path = "model.json"
    model = xgb.XGBRegressor(random_state=42)
    model.load_model(model_path)

    # Load label encoder
    run["training/label_encoder"].download()
    lbl_encoder_path = "label_encoder.joblib"
    lbl = joblib.load(lbl_encoder_path)
    X_valid = lbl.transform(X_valid)

    # Calculate scores
    model_score = model.score(X_valid, y_valid)
    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    mae = mean_absolute_error(y_valid, y_pred)

    # # # (neptune) Log scores
    run["training/val/r2"] = model_score
    run["training/val/rmse"] = rmse
    run["training/val/mae"] = mae

    # Visualize predictions
    sns.set()
    plt.rcParams["figure.figsize"] = 15, 8
    plt.rcParams["image.cmap"] = "viridis"
    plt.ioff()

    df_result = pd.DataFrame(
        data={
            "y_valid": y_valid.values,
            "y_pred": y_pred,
            "Week": valid_df.loc[valid_df.index].Week,
        },
        index=valid_df.index,
    )
    df_result = df_result.set_index("Week")

    plt.figure()
    preds_plot = sns.lineplot(data=df_result)

    # (neptune) Log predictions visualizations
    run["training/plots/ypred_vs_y_valid"].upload(File.as_image(preds_plot.figure))


if __name__ == "__main__":
    main()
