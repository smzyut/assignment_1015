from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


data_folder = Path(__file__).resolve().parent
output_folder = data_folder.parent / "out" / "modeling_submission"
output_folder.mkdir(parents=True, exist_ok=True)
file_names = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
load_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
lag_hours_list = [1, 3, 6, 12, 24, 48]
horizons = [1, 6, 24]
main_model = "HGB_OT_load_history"
validation_periods = [
    (pd.Timestamp("2017-04-01"), pd.Timestamp("2017-07-01")),
    (pd.Timestamp("2017-07-01"), pd.Timestamp("2017-10-01")),
    (pd.Timestamp("2017-10-01"), pd.Timestamp("2018-01-01")),
]
test_start = pd.Timestamp("2018-01-01")


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=10.0,
        early_stopping=False, random_state=42)


def predict_hgb(training_data, evaluation_data, columns):
    model = make_model()
    target_change = training_data["future_OT"] - training_data["OT_now"]
    model.fit(training_data[columns], target_change)
    predicted_change = model.predict(evaluation_data[columns])
    return evaluation_data["OT_now"].to_numpy() + predicted_change


def measure(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    hot = actual >= np.quantile(actual, 0.90)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "hot_count": int(hot.sum()),
        "hot_MAE": mean_absolute_error(actual[hot], predicted[hot]),
        "hot_underprediction": float(np.mean(actual[hot] - predicted[hot])),
    }


validation_rows = []
test_rows = []
prediction_tables = []

for file_name in file_names:
    print(f"========== {file_name} ==========", flush=True)
    df = pd.read_csv(data_folder / f"{file_name}.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    rows_per_hour = 4 if file_name.startswith("ETTm") else 1
    common = pd.DataFrame(index=df.index)
    common["OT_now"] = df["OT"]
    if rows_per_hour == 4:
        common["OT_previous_step"] = df["OT"].shift(1)
    for lag_hours in lag_hours_list:
        common[f"OT_{lag_hours}h_ago"] = df["OT"].shift(lag_hours * rows_per_hour)
    common["OT_change_1h"] = df["OT"] - df["OT"].shift(rows_per_hour)
    common["OT_change_24h"] = df["OT"] - df["OT"].shift(24 * rows_per_hour)
    ot_history_features = list(common.columns)
    common["OT_mean_6h"] = df["OT"].rolling(6 * rows_per_hour).mean()
    common["OT_mean_24h"] = df["OT"].rolling(24 * rows_per_hour).mean()
    ot_mean_features = ["OT_mean_6h", "OT_mean_24h"]

    load_now_features = []
    load_lag_features = []
    load_mean_features = []
    for column in load_columns:
        common[column] = df[column]
        load_now_features.append(column)
        for lag_hours in lag_hours_list:
            feature_name = f"{column}_{lag_hours}h_ago"
            common[feature_name] = df[column].shift(lag_hours * rows_per_hour)
            load_lag_features.append(feature_name)
        for window_hours in [6, 24]:
            feature_name = f"{column}_mean_{window_hours}h"
            common[feature_name] = df[column].rolling(window_hours * rows_per_hour).mean()
            load_mean_features.append(feature_name)

    for horizon_hours in horizons:
        data = common.copy()
        target_time = df.index + pd.Timedelta(hours=horizon_hours)
        data["target_time"] = target_time
        data["future_OT"] = df["OT"].shift(-horizon_hours * rows_per_hour)
        data["OT_target_previous_day"] = df["OT"].shift((24 - horizon_hours) * rows_per_hour)
        data["OT_target_two_days_ago"] = df["OT"].shift((48 - horizon_hours) * rows_per_hour)
        hour = target_time.hour + target_time.minute / 60
        day_of_year = target_time.dayofyear - 1
        days_in_year = np.where(target_time.is_leap_year, 366, 365)
        day_of_week = target_time.dayofweek
        data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        data["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
        data["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)
        data["weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        data["weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        target_features = ["OT_target_previous_day", "OT_target_two_days_ago",
            "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos", "weekday_sin", "weekday_cos"]
        ot_features = ot_history_features + ot_mean_features + target_features
        model_features = {
            "HGB_OT_no_mean": ot_history_features + target_features,
            "HGB_OT": ot_features,
            "HGB_OT_load_now": ot_features + load_now_features,
            "HGB_OT_load_lags": ot_features + load_now_features + load_lag_features,
            "HGB_OT_load_history": ot_features + load_now_features + load_lag_features + load_mean_features,
        }
        data = data.dropna()

        for fold_number, (start, end) in enumerate(validation_periods, start=1):
            training = data[data["target_time"] < start]
            validation = data[(data.index >= start) & (data["target_time"] < end)]
            if training.empty or validation.empty:
                raise ValueError(f"{file_name}: fold {fold_number} が空です")
            validation_predictions = {
                "persistence": validation["OT_now"].to_numpy(),
                "previous_day": validation["OT_target_previous_day"].to_numpy(),
            }
            for model_name, columns in model_features.items():
                validation_predictions[model_name] = predict_hgb(training, validation, columns)
            for model_name, predicted in validation_predictions.items():
                score = measure(validation["future_OT"], predicted)
                validation_rows.append({"dataset": file_name, "horizon_hours": horizon_hours,
                    "fold": fold_number, "train_end": str(start), "validation_end": str(end),
                    "model": model_name, "feature_count": len(model_features.get(model_name, [])),
                    "train_count": len(training), "validation_count": len(validation), **score})

        final_training = data[data["target_time"] < test_start]
        test = data[data.index >= test_start]
        if final_training.empty or test.empty:
            raise ValueError(f"{file_name}: 最終学習またはテスト期間が空です")
        test_predictions = {
            "persistence": test["OT_now"].to_numpy(),
            "previous_day": test["OT_target_previous_day"].to_numpy(),
            "HGB_OT": predict_hgb(final_training, test, ot_features),
            main_model: predict_hgb(final_training, test, model_features[main_model]),
        }
        for model_name, predicted in test_predictions.items():
            score = measure(test["future_OT"], predicted)
            test_rows.append({"dataset": file_name, "horizon_hours": horizon_hours,
                "model": model_name, "is_main_model": model_name == main_model,
                "train_count": len(final_training), "test_count": len(test), **score})
        score = measure(test["future_OT"], test_predictions[main_model])
        print(f"{horizon_hours:>2}時間先: {main_model} MAE={score['MAE']:.3f}", flush=True)
        prediction_tables.append(pd.DataFrame({
            "dataset": file_name, "horizon_hours": horizon_hours, "origin_time": test.index,
            "target_time": test["target_time"].to_numpy(), "actual_OT": test["future_OT"].to_numpy(),
            "main_model": main_model, "hgb_prediction": test_predictions[main_model],
            "ot_only_prediction": test_predictions["HGB_OT"],
            "persistence_prediction": test_predictions["persistence"],
            "previous_day_prediction": test_predictions["previous_day"],
        }))

validation_table = pd.DataFrame(validation_rows)
summary = validation_table.groupby(["dataset", "horizon_hours", "model"], as_index=False)[
    ["feature_count", "MAE", "RMSE", "hot_MAE", "hot_underprediction"]].mean()
summary = summary.rename(columns={"MAE": "mean_MAE", "RMSE": "mean_RMSE",
    "hot_MAE": "mean_hot_MAE", "hot_underprediction": "mean_hot_underprediction"})
print("\n2017年の3期間平均MAE（小さいほど良い）:")
print(summary.pivot(index=["dataset", "horizon_hours"], columns="model", values="mean_MAE").round(3).to_string())
print("\n2018年テストMAE（小さいほど良い）:")
test_table = pd.DataFrame(test_rows)
print(test_table.pivot(index=["dataset", "horizon_hours"], columns="model", values="MAE").round(3).to_string())
print("\nETTh2・6時間先の高温域とRMSE:")
print(test_table[(test_table["dataset"] == "ETTh2")
                 & (test_table["horizon_hours"] == 6)][
    ["model", "test_count", "RMSE", "hot_count", "hot_MAE", "hot_underprediction"]].round(3).to_string(index=False))
all_predictions = pd.concat(prediction_tables, ignore_index=True)

for file_name in file_names:
    for horizon_hours in horizons:
        one_result = all_predictions[(all_predictions["dataset"] == file_name)
            & (all_predictions["horizon_hours"] == horizon_hours)].sort_values("target_time")
        plot_columns = ["actual_OT", "hgb_prediction"]
        daily = one_result.set_index("target_time")[plot_columns].resample("D").mean()
        first_14_days = one_result[one_result["target_time"] < test_start + pd.Timedelta(days=14)]
        peak_time = one_result.loc[one_result["actual_OT"].idxmax(), "target_time"]
        peak_week = one_result[(one_result["target_time"] >= peak_time - pd.Timedelta(days=3))
            & (one_result["target_time"] <= peak_time + pd.Timedelta(days=3))]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for ax, table, title in [(axes[0], daily.reset_index(), "Whole test period: daily mean"),
            (axes[1], first_14_days, "First 14 test days: original resolution"),
            (axes[2], peak_week, f"Around highest actual OT ({peak_time:%Y-%m-%d})")]:
            ax.plot(table["target_time"], table["actual_OT"], label="Actual OT", linewidth=1.2)
            ax.plot(table["target_time"], table["hgb_prediction"], label="HGB: OT + load history", linewidth=1.2)
            ax.set_title(title)
            ax.set_ylabel("OT")
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", labelrotation=25)
        fig.suptitle(f"{file_name}: {horizon_hours}h ahead")
        fig.tight_layout()
        fig.savefig(output_folder / f"{file_name}_{horizon_hours}h_prediction_vs_actual.png", dpi=150)
        plt.close(fig)
print("予測図を保存しました:", output_folder)
