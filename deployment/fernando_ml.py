import argparse
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# --- CLI Arguments Setup ---
parser = argparse.ArgumentParser(description="Train and evaluate wind power forecasting pipeline.")
parser.add_argument(
    "--data-path",
    type=str,
    default="data",
    help="Directory containing the CSV datasets",
)
args = parser.parse_args()

data_dir = args.data_path
power_path = os.path.join(data_dir, "power.csv")
weather_path = os.path.join(data_dir, "weather.csv")
future_path = os.path.join(data_dir, "future.csv")


def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def create_eda_plots(joined_dfs):
    """Create exploratory data analysis plots for wind power data."""
    fig, ax = plt.subplots(1, 3, figsize=(25, 4))

    # Speed and Power for the last 7 days
    ax[0].plot(joined_dfs["Speed"].tail(int(7 * 24 / 3)), label="Speed", color="blue")
    ax[0].plot(joined_dfs["Total"].tail(int(7 * 24 / 3)), label="Power", color="tab:red")
    ax[0].set_title("Windspeed & Power Generation over last 7 days")
    ax[0].set_xlabel("Time")
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[0].set_ylabel("Windspeed [m/s], Power [MW]")
    ax[0].legend()

    # Speed vs Total (Power Curve nature)
    ax[1].scatter(joined_dfs["Speed"], joined_dfs["Total"])
    power_curve = joined_dfs.groupby("Speed").median(numeric_only=True)["Total"]
    ax[1].plot(power_curve.index, power_curve.values, "k:", label="Power Curve")
    ax[1].legend()
    ax[1].set_title("Windspeed vs Power")
    ax[1].set_ylabel("Power [MW]")
    ax[1].set_xlabel("Windspeed [m/s]")

    # Speed and Power per Wind Direction
    if "Direction" in joined_dfs.columns:
        wind_grouped_by_direction = joined_dfs.groupby("Direction").mean(numeric_only=True).reset_index()
        bar_width = 0.5
        x = np.arange(len(wind_grouped_by_direction.index))
        ax[2].bar(x, wind_grouped_by_direction.Total, width=0.5, label="Power", color="tab:red")
        ax[2].bar(x + bar_width, wind_grouped_by_direction.Speed, width=0.5, label="Speed", color="blue")
        ax[2].legend()
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(wind_grouped_by_direction.Direction)
        ax[2].tick_params(axis="x", labelrotation=45)
        ax[2].set_title("Speed and Power per Direction")
    else:
        ax[2].axis("off")

    plt.tight_layout()
    return fig


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
if not os.environ.get("MLFLOW_RUN_ID"):
    mlflow.set_experiment("orkney_wind_power")

REGISTERED_MODEL_NAME = "OrkneyWindForecastModel"
MODEL_LABEL = "XGBoost"

with mlflow.start_run(run_name=MODEL_LABEL, nested=True):
    print(f"Loading data from {data_dir}...")
    power_df = read_csv_with_time_index(power_path)
    wind_df = read_csv_with_time_index(weather_path)

    print("Starting preprocessing...")
    # Resample to 3h average, interpolate brief gaps, align indices
    joined_dfs = power_df[["Total"]].resample("3h").mean()
    joined_dfs["Total"] = joined_dfs["Total"].interpolate(method="time", limit=2)
    joined_dfs = joined_dfs.dropna(subset=["Total"])
    joined_dfs = joined_dfs.join(wind_df, how="inner")
    joined_dfs = joined_dfs.drop(columns=["Lead_hours", "Source_time"], errors="ignore")
    joined_dfs = joined_dfs.sort_index()

    # Save and log EDA plots
    os.makedirs("plots", exist_ok=True)
    eda_fig = create_eda_plots(joined_dfs)
    eda_fig.savefig("plots/eda_plots.png")
    mlflow.log_artifact("plots/eda_plots.png")
    plt.close(eda_fig)

    # Feature and target definition
    X = joined_dfs[["Direction", "Speed"]]
    y = joined_dfs["Total"]

    # Temporal 80/20 train-test split
    index_split = int(len(joined_dfs) * 0.8)
    X_train, X_test = X.iloc[:index_split], X.iloc[index_split:]
    y_train, y_test = y.iloc[:index_split], y.iloc[index_split:]

    # Pipeline definition
    tree_processor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Direction"]),
            ("num", "passthrough", ["Speed"]),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", tree_processor),
        ("regressor", XGBRegressor(n_estimators=100, random_state=42)),
    ])

    print("Starting 5-fold TimeSeries cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds_val = pipeline.predict(X_val)
        cv_mae_scores.append(mean_absolute_error(y_val, preds_val))

    mean_mae = float(np.mean(cv_mae_scores))
    std_mae = float(np.std(cv_mae_scores))

    mlflow.log_metric("Mean_CV_MAE", mean_mae)
    mlflow.log_metric("Std_CV_MAE", std_mae)
    print(f"{MODEL_LABEL} -> Mean CV MAE: {mean_mae:.2f} MW (+/- {std_mae:.2f})")

    print("Fitting final model on complete training set...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    predictions = pipeline.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, predictions))
    test_rmse = float(root_mean_squared_error(y_test, predictions))

    mlflow.log_metric("Test_MAE", test_mae)
    mlflow.log_metric("Test_RMSE", test_rmse)

    # Log and register model with an inferred signature for serving
    signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        signature=signature,
        input_example=X_train.head(1),
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBRegressor'],
    )

    # Plot test predictions vs ground truth
    plt.figure(figsize=(15, 4))
    plt.plot(np.arange(len(predictions)), predictions, label="Predictions")
    plt.plot(np.arange(len(y_test)), y_test.values, label="Truth")
    plt.title("XGBoost Predictions vs Ground Truth")
    plt.legend()
    plt.savefig("plots/predictions.png")
    plt.close()
    mlflow.log_artifact("plots/predictions.png")

    # Generate predictions on future.csv (per assignment criteria)
    if os.path.exists(future_path):
        future_df = read_csv_with_time_index(future_path)
        future_X = future_df[["Direction", "Speed"]]
        future_preds = pipeline.predict(future_X)
        future_df["Predicted_Total"] = future_preds
        future_df.to_csv("plots/future_predictions.csv")
        mlflow.log_artifact("plots/future_predictions.csv")
        print("Future forecast generated and logged successfully.")
    else:
        print(f"Warning: {future_path} not found. Skipping future predictions.")

    print(f"Workflow complete. Registered model: '{REGISTERED_MODEL_NAME}'.")