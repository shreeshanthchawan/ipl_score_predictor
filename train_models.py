"""Train models for the ES335 IPL next-score prediction study.

Each supervised example uses a batter's five previous batting-appearance scores
to predict the runs scored in the next batting appearance.

Run from this folder:
    python train_models.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

DELIVERIES_FILE = ROOT / "deliveries_updated_ipl_upto_2025.csv"
FEATURES = [f"match_{i}" for i in range(1, 6)]  # oldest to most recent


def load_player_match_scores():
    deliveries = pd.read_csv(DELIVERIES_FILE)
    original_rows = len(deliveries)
    deliveries["date"] = pd.to_datetime(deliveries["date"], errors="coerce")
    deliveries["batsman_runs"] = pd.to_numeric(
        deliveries["batsman_runs"], errors="coerce"
    )

    quality = [
        ("Original delivery rows", original_rows),
        ("Exact duplicate rows", int(deliveries.duplicated().sum())),
        ("Rows missing match ID", int(deliveries["matchId"].isna().sum())),
        ("Rows missing batter name", int(deliveries["batsman"].isna().sum())),
        ("Rows missing date", int(deliveries["date"].isna().sum())),
        ("Rows missing batsman runs", int(deliveries["batsman_runs"].isna().sum())),
    ]
    pd.DataFrame(quality, columns=["check", "result"]).to_csv(
        OUTPUT / "data_quality_checks.csv", index=False
    )

    deliveries = deliveries.drop_duplicates()
    deliveries = deliveries.dropna(subset=["matchId", "date", "batsman"]).copy()
    deliveries["batsman_runs"] = deliveries["batsman_runs"].fillna(0)

    # Sum batter runs over all deliveries in each match.
    player_matches = (
        deliveries.groupby(["matchId", "date", "batsman"], as_index=False)
        .agg(runs=("batsman_runs", "sum"))
        .sort_values(["batsman", "date", "matchId"])
        .reset_index(drop=True)
    )

    # Shift by one appearance so the target match's score is never an input.
    for shift in range(1, 6):
        player_matches[f"match_{6 - shift}"] = (
            player_matches.groupby("batsman")["runs"].shift(shift)
        )

    return player_matches


def export_browser_models(linear_model, forest_model):
    scaler = linear_model.named_steps["standardscaler"]
    regression = linear_model.named_steps["linearregression"]
    linear_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": regression.coef_.tolist(),
        "intercept": float(regression.intercept_),
    }

    trees = []
    for estimator in forest_model.estimators_:
        tree = estimator.tree_
        trees.append({
            "left": tree.children_left.tolist(),
            "right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist(),
        })

    forest_data = {"trees": trees, "n_features": 5}
    with open(OUTPUT / "linear_model.json", "w", encoding="utf-8") as file:
        json.dump(linear_data, file)
    with open(OUTPUT / "random_forest_model.json", "w", encoding="utf-8") as file:
        json.dump(forest_data, file, separators=(",", ":"))


def main():
    player_matches = load_player_match_scores()
    data = player_matches.dropna(subset=FEATURES).copy()
    player_matches.to_csv(OUTPUT / "player_match_scores.csv", index=False)
    data.to_csv(OUTPUT / "modeling_dataset.csv", index=False)

    # Keep later dates out of training to mimic predicting future appearances.
    dates = np.sort(data["date"].unique())
    split_date = dates[int(len(dates) * 0.80)]
    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(
            StandardScaler(), LinearRegression()
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    test_output = test[["matchId", "date", "batsman"] + FEATURES + ["runs"]].copy()
    for name, model in models.items():
        model.fit(train[FEATURES], train["runs"])
        predictions = model.predict(test[FEATURES])
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(test["runs"], predictions),
            "RMSE": mean_squared_error(test["runs"], predictions) ** 0.5,
            "R2": r2_score(test["runs"], predictions),
        })
        test_output[name.lower().replace(" ", "_") + "_prediction"] = predictions

    metrics = pd.DataFrame(results)
    metrics.to_csv(OUTPUT / "model_metrics.csv", index=False)
    test_output.to_csv(OUTPUT / "test_predictions.csv", index=False)

    # Save a short audit of the largest Random Forest misses.
    test_output["absolute_error_rf"] = (
        test_output["runs"] - test_output["random_forest_prediction"]
    ).abs()
    test_output.sort_values("absolute_error_rf", ascending=False).head(20).to_csv(
        OUTPUT / "largest_prediction_errors.csv", index=False
    )

    plt.figure(figsize=(7, 5))
    plt.bar(metrics["Model"], metrics["MAE"])
    plt.ylabel("Mean absolute error (runs)")
    plt.title("Test-set prediction error")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "model_mae.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(test["runs"], models["Random Forest"].predict(test[FEATURES]), alpha=0.2)
    plt.xlabel("Actual next-appearance runs")
    plt.ylabel("Random Forest prediction")
    plt.title("Random Forest: actual vs predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT / "actual_vs_predicted.png", dpi=150)
    plt.close()

    joblib.dump(models["Linear Regression"], OUTPUT / "linear_regression.joblib")
    joblib.dump(models["Random Forest"], OUTPUT / "random_forest.joblib")
    export_browser_models(models["Linear Regression"], models["Random Forest"])

    # Save recent-five-match history for the website's player autofill.
    player_lookup = {}
    for batter, history in player_matches.groupby("batsman"):
        history = history.sort_values(["date", "matchId"])
        last_five = history.tail(5)
        if len(last_five) == 5:
            player_lookup[batter] = {
                "scores": last_five["runs"].astype(int).tolist(),
                "dates": [date.strftime("%Y-%m-%d") for date in last_five["date"]],
                "last_appearance": history["date"].max().strftime("%Y-%m-%d"),
            }

    metadata = {
        "player_match_rows": int(len(player_matches)),
        "supervised_rows": int(len(data)),
        "players_with_five_appearances": int(len(player_lookup)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": train["date"].max().strftime("%Y-%m-%d"),
        "test_start": test["date"].min().strftime("%Y-%m-%d"),
        "test_end": test["date"].max().strftime("%Y-%m-%d"),
        "feature_order": FEATURES,
    }
    site_data = {
        "players": player_lookup,
        "metrics": metrics.to_dict(orient="records"),
        "meta": metadata,
    }
    site = ROOT / "site"
    site.mkdir(exist_ok=True)
    with open(site / "data.json", "w", encoding="utf-8") as file:
        json.dump(site_data, file, separators=(",", ":"))
    with open(site / "models.json", "w", encoding="utf-8") as file:
        json.dump({
            "linear": json.load(open(OUTPUT / "linear_model.json", encoding="utf-8")),
            "forest": json.load(open(OUTPUT / "random_forest_model.json", encoding="utf-8")),
        }, file, separators=(",", ":"))

    print(metrics.to_string(index=False))
    print(f"\nTraining rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Test period: {metadata['test_start']} to {metadata['test_end']}")


if __name__ == "__main__":
    main()
