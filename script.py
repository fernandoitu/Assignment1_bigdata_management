########################################################################################################################
# IMPORTS
# You absolutely need these
import mlflow
import os

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import skops.io as sio

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###

########################################################################################################################

## Step 1: The Data (from CSVs)


def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def create_eda_plots(joined_dfs):
    """
    Create exploratory data analysis plots for wind power data
    """
    fig, ax = plt.subplots(1,3, figsize=(25,4))

    # Speed and Power for the last 7 days
    ax[0].plot(joined_dfs["Speed"].tail(int(7*24/3)), label="Speed", color="blue")
    ax[0].plot(joined_dfs["Total"].tail(int(7*24/3)), label="Power", color="tab:red")
    ax[0].set_title("Windspeed & Power Generation over last 7 days")
    ax[0].set_xlabel("Time")
    ax[0].tick_params(axis='x', labelrotation = 45)
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
        ax[2].tick_params(axis='x', labelrotation = 45)
        ax[2].set_title("Speed and Power per Direction")
    else:
        ax[2].axis("off")

    plt.tight_layout()
    return fig

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000") # We set the MLFlow UI to display in our local host.

mlflow.set_experiment("template-model")

# Start a run
with mlflow.start_run(run_name="LinearRegression"):

    print("Loading data")

    # --- Load from CSVs ---
    power_df = read_csv_with_time_index("data/power.csv")
    wind_df = read_csv_with_time_index("data/weather.csv")

    print("Starting preprocessing")

    # --- Join datasets (as before) ---
    joined_dfs = power_df.join(wind_df, how="inner").dropna(subset=["Total", "Speed"])

    # --- Create and save EDA plots ---
    os.makedirs("plots", exist_ok=True)
    eda_fig = create_eda_plots(joined_dfs)
    eda_fig.savefig("plots/eda_plots.png")
    mlflow.log_artifact("plots/eda_plots.png")
    plt.close(eda_fig)

    # --- Model section (same as before) ---
    def load_and_predict_model(model_name, model_version, new_data):
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        return model.predict(new_data)

    X = joined_dfs[["Speed"]]
    y = joined_dfs["Total"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    print("Starting training")

    # Train and evaluate model
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Plot predictions
    plt.figure(figsize=(15, 4))
    plt.plot(np.arange(len(predictions)), predictions, label="Predictions")
    plt.plot(np.arange(len(y_test)), y_test, label="Truth")
    plt.legend()
    plt.savefig(f"plots/predictions.png")
    plt.close()
    mlflow.log_artifact(f"plots/predictions.png")

    # No need to manually log metrics - autologging handles:
    # - Parameters
    # - Metrics (R², MSE, MAE)
    # - Model artifacts
    # - Model signature
    # - Feature importance (for supported models)


    

########################################################################################################################
