import argparse
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

# --- CLI Arguments Setup ---
parser = argparse.ArgumentParser(description="Train, compare and register wind power forecasting pipelines.")
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

# --- Pipeline settings (all of them are logged as MLflow parameters) ---
RESAMPLE_FREQ = "3h"
INTERPOLATION_LIMIT = 2  # max. consecutive 3h blocks (6 hours) that are filled by interpolation
TRAIN_FRACTION = 0.8
N_SPLITS = 5
SEED = 42

# The 16 compass points in clockwise order, so that index * 22.5 degrees is the meteorological angle
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def index_to_sin_cos(idx):
    """Map a compass index (0..15) to a point on the unit circle: (sin, cos) of its angle in radians."""
    angle = idx * 2 * np.pi / len(COMPASS)
    return np.hstack([np.sin(angle), np.cos(angle)])


def build_processors():
    """Preprocessing for the linear model (scaled speed) and for the tree models (unscaled speed)."""
    # Direction: label -> index (unknown or missing -> NaN) -> impute -> radians -> (sin, cos)
    def direction_pipe():
        return Pipeline([
            ("ordinal", OrdinalEncoder(categories=[COMPASS],
                                       handle_unknown="use_encoded_value",
                                       unknown_value=np.nan)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("sincos", FunctionTransformer(index_to_sin_cos)),
        ])

    linear_processor = ColumnTransformer([
        ("dir", direction_pipe(), ["Direction"]),
        ("speed", Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())]), ["Speed"]),
    ])
    tree_processor = ColumnTransformer([
        ("dir", direction_pipe(), ["Direction"]),
        ("speed", SimpleImputer(strategy="median"), ["Speed"]),
    ])
    return linear_processor, tree_processor


def evaluate(y_true, y_pred):
    """The same three regression metrics for every model."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


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
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
if not os.environ.get("MLFLOW_RUN_ID"):
    mlflow.set_experiment("orkney_wind_power")

REGISTERED_MODEL_NAME = "OrkneyWindForecastModel"

# The parent run holds the shared settings and artifacts; every model gets its own nested run.
with mlflow.start_run(run_name="model_comparison"):
    print(f"Loading data from {data_dir}...")
    power_df = read_csv_with_time_index(power_path)
    wind_df = read_csv_with_time_index(weather_path)

    print("Starting preprocessing...")
    # Resample to 3h average, interpolate brief gaps, align indices
    joined_dfs = power_df[["Total"]].resample(RESAMPLE_FREQ).mean()
    joined_dfs["Total"] = joined_dfs["Total"].interpolate(method="time", limit=INTERPOLATION_LIMIT)
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

    # Temporal train/test split (the test set is the latest part of the time series)
    index_split = int(len(joined_dfs) * TRAIN_FRACTION)
    X_train, X_test = X.iloc[:index_split], X.iloc[index_split:]
    y_train, y_test = y.iloc[:index_split], y.iloc[index_split:]

    mlflow.log_params({
        "data_path": data_dir,
        "resample_freq": RESAMPLE_FREQ,
        "interpolation_limit": INTERPOLATION_LIMIT,
        "train_fraction": TRAIN_FRACTION,
        "n_splits": N_SPLITS,
        "n_rows": len(joined_dfs),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "selection_metric": "Mean_CV_MAE",
    })

    linear_processor, tree_processor = build_processors()
    models = [
        ("Ridge", Ridge(), linear_processor),
        ("RandomForest", RandomForestRegressor(random_state=SEED), tree_processor),
        ("XGBoost", XGBRegressor(random_state=SEED), tree_processor),
    ]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = {}

    for model_name, regressor, processor in models:
        pipeline = Pipeline([("preprocessor", processor), ("regressor", regressor)])

        with mlflow.start_run(run_name=model_name, nested=True):
            print(f"Training {model_name}...")
            mlflow.log_param("model", model_name)
            mlflow.log_params({
                k: v for k, v in regressor.get_params().items()
                if isinstance(v, (int, float, str, bool, type(None)))
            })

            # 5-fold temporal cross-validation (each fold trains on the past and validates on the next block)
            cv_scores = {"MAE": [], "RMSE": [], "R2": []}
            for train_idx, val_idx in tscv.split(X):
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                fold_scores = evaluate(y.iloc[val_idx], pipeline.predict(X.iloc[val_idx]))
                for metric, value in fold_scores.items():
                    cv_scores[metric].append(value)

            for metric, values in cv_scores.items():
                mlflow.log_metric(f"Mean_CV_{metric}", float(np.mean(values)))
                mlflow.log_metric(f"Std_CV_{metric}", float(np.std(values)))

            # Final model: fit on the whole training split, evaluate once on the held-out test split
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            test_scores = evaluate(y_test, predictions)
            for metric, value in test_scores.items():
                mlflow.log_metric(f"Test_{metric}", value)

            # The preprocessing contains a custom function, so the model is stored with cloudpickle
            signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(1),
                serialization_format="cloudpickle",
            )

            results[model_name] = {
                "pipeline": pipeline,
                "predictions": predictions,
                "model_uri": model_info.model_uri,
                "cv_mae": float(np.mean(cv_scores["MAE"])),
                "cv_mae_std": float(np.std(cv_scores["MAE"])),
                "cv_rmse": float(np.mean(cv_scores["RMSE"])),
                "cv_r2": float(np.mean(cv_scores["R2"])),
                **{f"test_{k.lower()}": v for k, v in test_scores.items()},
            }

    # Compare the models. The best one is selected by cross-validated MAE (validation folds),
    # the test split is kept as an independent check of the chosen model.
    print(f"\n{'model':<13} {'CV MAE':>14} {'CV RMSE':>8} {'CV R2':>7} | {'Test MAE':>8} {'Test RMSE':>9} {'Test R2':>8}")
    for name, r in results.items():
        print(f"{name:<13} {r['cv_mae']:>7.2f} +/- {r['cv_mae_std']:.2f} {r['cv_rmse']:>8.2f} {r['cv_r2']:>7.3f} | "
              f"{r['test_mae']:>8.2f} {r['test_rmse']:>9.2f} {r['test_r2']:>8.3f}")

    best_name = min(results, key=lambda n: results[n]["cv_mae"])
    best = results[best_name]
    print(f"\nBest model: {best_name} (lowest mean CV MAE: {best['cv_mae']:.2f} MW)")
    mlflow.set_tag("best_model", best_name)

    # Register the best model in the MLflow Model Registry
    model_version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
    MlflowClient().set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, "model_type", best_name)
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {model_version.version} ({best_name}).")

    # Plot test predictions vs ground truth for the best model
    plt.figure(figsize=(15, 4))
    plt.plot(y_test.index, best["predictions"], label="Predictions")
    plt.plot(y_test.index, y_test.values, label="Truth")
    plt.title(f"{best_name} Predictions vs Ground Truth (test set)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/predictions.png")
    plt.close()
    mlflow.log_artifact("plots/predictions.png")

    # Generate predictions on future.csv with the best model (per assignment criteria)
    if os.path.exists(future_path):
        future_df = read_csv_with_time_index(future_path)
        future_X = future_df[["Direction", "Speed"]]
        future_df["Predicted_Power"] = best["pipeline"].predict(future_X)
        future_df.to_csv("plots/future_predictions.csv")
        mlflow.log_artifact("plots/future_predictions.csv")
        print("First 10 predictions on future.csv:")
        print(future_df[["Direction", "Speed", "Predicted_Power"]].head(10).to_string())
    else:
        print(f"Warning: {future_path} not found. Skipping future predictions.")

    print(f"Workflow complete. Registered model: '{REGISTERED_MODEL_NAME}'.")
