import argparse
import os

import joblib
import neptune
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from neptune.integrations.xgboost import NeptuneCallback
from sklearn import preprocessing


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
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--valid_data", type=str, help="path to validation data")
    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))

    print(train_df)

    # Split data into train and validation
    features_to_exclude = ["Weekly_Sales", "Date", "Year"]
    X = train_df.loc[:, ~train_df.columns.isin(features_to_exclude)]
    y = train_df.loc[:, "Weekly_Sales"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=False
    )

    # Encoding data to categorical variables
    lbl = preprocessing.OneHotEncoder(handle_unknown="ignore")
    X_train = lbl.fit_transform(X_train)

    # (neptune) Initialize Neptune run
    run = neptune.init_run(
        tags=["MLOps", "baseline", "xgboost", "walmart-sales"],
        name="XGBoost",
    )

    neptune_callback = NeptuneCallback(run=run)

    #  Train model
    model = xgb.XGBRegressor(callbacks=[neptune_callback], random_state=42).fit(X_train, y_train)

    # Save model
    model_filename = "model.json"
    model.save_model(model_filename)
    run["training/model"].upload(model_filename)

    # Save Label encoder
    lbl_encoder_filename = "label_encoder.joblib"
    joblib.dump(lbl, lbl_encoder_filename)
    run["training/label_encoder"].upload(lbl_encoder_filename)

    # Concatenate x and y train data
    validation_df = pd.concat([X_val, y_val], axis=1)

    # Save train and validation data
    validation_data_path = os.path.join(args.valid_data, "validation_data.csv")
    validation_df.to_csv(validation_data_path, index=False)

    print(f"df encoded: {validation_df}")


if __name__ == "__main__":
    main()
