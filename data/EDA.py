import pandas as pd
import matplotlib

matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pathlib import Path

file_names = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
load_columns = ["HUFL","HULL","MUFL","MULL","LUFL","LULL"]
numeric_columns = load_columns + ["OT"]
test_start = pd.Timestamp("2018-01-01")
sns.set_theme(style="whitegrid")
data_folder = Path(__file__).resolve().parent
output_folder = data_folder.parent / "out" / "eda_submission"
output_folder.mkdir(parents=True, exist_ok=True)


feature_results = []
baseline_results = []

for file_name in file_names:
    print(f" ===={file_name}====")
    full_df = pd.read_csv(data_folder / f"{file_name}.csv",parse_dates=["date"])
    full_df = full_df.set_index("date").sort_index()
    df = full_df[full_df.index < test_start].copy()
    rows_per_hour = 4 if file_name.startswith("ETTm") else 1


    fig,axes = plt.subplots(1,2,figsize=(12,4))
    sns.histplot(data=df,x="OT",bins=50,ax=axes[0])
    axes[0].set_title(f"{file_name}:OT_distribution")
    daily_OT = df["OT"].resample("D").mean()
    sns.lineplot(x=daily_OT.index,y=daily_OT.values,ax=axes[1])
    axes[1].set_title(f"{file_name}:daily_mean_OT")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_folder / f"{file_name}_distribution_time.png",dpi=150)
    plt.close(fig)

    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek

    daily_mean_at_each_row = df["OT"].groupby(df.index.normalize()).transform("mean")
    df["OT_minus_daily_mean"] = df["OT"] - daily_mean_at_each_row

    week_start = df.index.normalize() - pd.to_timedelta(df.index.dayofweek,unit="D")
    weekly_mean_at_each_row = df["OT"].groupby(week_start).transform("mean")
    df["OT_minus_weekly_mean"] = df["OT"] - weekly_mean_at_each_row
    monthly = df["OT"].resample("MS").mean().reset_index()
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    fig,axes = plt.subplots(1,3,figsize=(15,4))
    sns.pointplot(data=df,x='hour',y="OT_minus_daily_mean",ax=axes[0],errorbar=None)
    axes[0].set_title("Hour: difference from daily_mean")
    sns.pointplot(data=df,x="weekday",y="OT_minus_weekly_mean",ax=axes[1],errorbar=None)
    axes[1].set_title("Weekday:difference from weekly mean")
    axes[1].set_xticks(range(7),["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    sns.lineplot(data=monthly,x="month",y="OT",hue="year",marker="o",ax=axes[2])
    axes[2].set_title("Monthly mean by year")
    axes[2].set_xticks(range(1,13))
    fig.suptitle(file_name)
    fig.tight_layout()
    fig.savefig(output_folder / f"{file_name}_calendar.png",dpi=150)
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(7,6))
    sns.heatmap(df[numeric_columns].corr(),annot=True,fmt=".2f",cmap="coolwarm",vmin=-1,vmax=1,ax=ax)
    ax.set_title(f"{file_name}:same-time correlations (description only)")
    fig.tight_layout()
    fig.savefig(output_folder / f"{file_name}_correlation.png",dpi=150)
    plt.close(fig)

    lag_24_hours = 24 * rows_per_hour
    fig,axes = plt.subplots(2,2,figsize=(12,7))
    plot_acf(df["OT"],lags=lag_24_hours,ax=axes[0,0],title="OT:ACF",fft=True)
    plot_pacf(df["OT"],lags=lag_24_hours,method="ywm",ax=axes[0,1],title="OT:PACF")
    OT_change = df["OT"].diff().dropna()
    plot_acf(OT_change,lags=lag_24_hours,ax=axes[1,0],title="OT change:ACF",fft=True)
    plot_pacf(OT_change,lags=lag_24_hours,method="ywm",ax=axes[1,1],title="OT change:PACF")
    for ax in axes.flat:
        ax.set_xlabel(f"Lag (rows;24hours = {lag_24_hours} rows)")
    fig.suptitle(file_name)
    fig.tight_layout()
    fig.savefig(output_folder / f"{file_name}_acf_pacf.png",dpi=150)
    plt.close(fig)

    one_dataset_lags = []
    for column in load_columns:
        for lag_hours in [1,3,6,12,24,48]:
            lag_rows = lag_hours * rows_per_hour
            past_load = df[column].shift(lag_rows)
            correlation = df["OT"].corr(past_load)
            changed_correlation = df["OT"].diff().corr(df[column].diff().shift(lag_rows))
            one_dataset_lags.append({"dataset":file_name,"column":column,"lag_hours":lag_hours,"corr":correlation,"changed_corr":changed_correlation})
    lag_table = pd.DataFrame(one_dataset_lags)

    fig,axes = plt.subplots(1,2,figsize=(13,4))
    sns.lineplot(data=lag_table,x="lag_hours",y="corr",hue="column",marker="o",ax=axes[0])
    axes[0].set_title("OT vs past_load")
    sns.lineplot(data=lag_table,x="lag_hours",y="changed_corr",hue="column",marker="o",ax=axes[1],legend=False)
    axes[1].set_title("First differences:same lags")
    fig.suptitle(file_name)
    fig.tight_layout()
    fig.savefig(output_folder / f"{file_name}_lagged_correlation.png",dpi=150)
    plt.close(fig)

    for horizon_hours in [1,6,24]:
        future_OT = df["OT"].shift(-horizon_hours * rows_per_hour)
        future_change = future_OT - df["OT"]
        one_horizon_features = []
        for column in ["OT"] + load_columns:
            candidates = {"now":df[column]}
            for lag_hours in [1,3,6,12,24,48]:
                candidates[f"lag_{lag_hours}h"] = df[column].shift(lag_hours * rows_per_hour)
            for window_hours in [6,24]:
                candidates[f"mean_{window_hours}h"] = df[column].rolling(window_hours * rows_per_hour).mean()
            for transform,feature in candidates.items():
                valid = feature.notna() & future_OT.notna()
                one_horizon_features.append({"dataset":file_name,"horizon_hours":horizon_hours,
                                             "column":column,"transform":transform,
                                             "feature":f"{column}_{transform}",
                                             "valid_count":int(valid.sum()),
                                             "corr_future_OT":feature[valid].corr(future_OT[valid]),
                                             "corr_future_change":feature[valid].corr(future_change[valid])})
        feature_results.extend(one_horizon_features)
        if horizon_hours == 6:
            feature_table = pd.DataFrame(one_horizon_features)
            load_table = feature_table[feature_table["column"].isin(load_columns)].copy()
            lag_table_for_plot = load_table[load_table["transform"].str.startswith("lag_")
                                            | load_table["transform"].eq("now")].copy()
            lag_table_for_plot["hours_ago"] = lag_table_for_plot["transform"].str.extract(r"lag_(\d+)h")[0].fillna(0).astype(int)
            mean_table_for_plot = load_table[load_table["transform"].str.startswith("mean_")].copy()
            mean_table_for_plot["window_hours"] = mean_table_for_plot["transform"].str.extract(r"mean_(\d+)h")[0].astype(int)
            fig,axes = plt.subplots(1,2,figsize=(13,4.6),sharey=True)
            sns.lineplot(data=lag_table_for_plot,x="hours_ago",y="corr_future_change",hue="column",marker="o",ax=axes[0])
            sns.lineplot(data=mean_table_for_plot,x="window_hours",y="corr_future_change",hue="column",marker="o",ax=axes[1],legend=False)
            axes[0].set_title("Current load and past lags")
            axes[0].set_xlabel("Hours before forecast origin (0 = now)")
            axes[1].set_title("Load moving averages up to origin")
            axes[1].set_xlabel("Past window length (hours)")
            axes[1].set_xticks([6,24])
            axes[0].set_ylabel("Correlation with OT change in next 6h")
            axes[1].set_ylabel("")
            axes[0].set_ylim(-0.4,0.4)
            handles,labels = axes[0].get_legend_handles_labels()
            axes[0].legend_.remove()
            fig.legend(handles,labels,loc="lower center",ncol=6,frameon=False)
            fig.suptitle(f"{file_name}: load features vs 6h future OT change")
            fig.tight_layout(rect=[0,0.07,1,1])
            fig.savefig(output_folder / f"{file_name}_load_feature_correlations_6h.png",dpi=150)
            plt.close(fig)

    rolling_table = pd.DataFrame(index=df.index)
    for column in load_columns:
        past_load = df[column].shift(6*rows_per_hour)
        rolling_table[column] = df["OT"].rolling(7*24*rows_per_hour).corr(past_load)
    rolling_daily = rolling_table.resample("D").last()

    fig,ax = plt.subplots(figsize=(12,4))
    
    sns.lineplot(data=rolling_daily,ax=ax)
    ax.set_title(f"{file_name}:7day rolling corr with load 6h earlier")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(output_folder / f"{file_name}_rolling_correlation.png",dpi=150)
    plt.close(fig)

    for horizon_hours in [1,6,24]:
        target = full_df["OT"].shift(-horizon_hours * rows_per_hour)
        persistence = full_df["OT"]
        previous_day = full_df["OT"].shift((24 - horizon_hours) * rows_per_hour)
        test_rows = full_df.index >= test_start
        for baseline_name,prediction in [("persistence",persistence),("previous_day",previous_day)]:
            error = target[test_rows] - prediction[test_rows]
            baseline_results.append({"dataset":file_name,"horizon_hours":horizon_hours,"baseline":baseline_name,"test_count":int(error.notna().sum()),"test_MAE": error.abs().mean()})
            


print("\n6時間先：各負荷の起点値・6時間ラグ・6/24時間平均と将来変化量の相関:")
feature_table = pd.DataFrame(feature_results)
selected = feature_table[(feature_table["horizon_hours"] == 6)
                         & feature_table["column"].isin(load_columns)
                         & feature_table["transform"].isin(["now","lag_6h","mean_6h","mean_24h"])]
print(selected.pivot(index=["dataset","column"],columns="transform",values="corr_future_change").round(3).to_string())
print("\n2018年テストの単純基準MAE:")
baseline_table = pd.DataFrame(baseline_results)
print(baseline_table.pivot(index=["dataset","horizon_hours"],columns="baseline",values="test_MAE").round(3).to_string())
print("\n図を保存しました:", output_folder)
