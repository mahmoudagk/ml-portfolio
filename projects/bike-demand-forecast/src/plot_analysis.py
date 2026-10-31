"""Generate extra descriptive and test-error figures for the bike-demand project."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import IMAGES, get_data


NAVY = "#18324B"
TEAL = "#139A9A"
GOLD = "#E9A33B"
CORAL = "#E56B6F"
MUTED = "#66788A"
GRID = "#DCE5ED"
BG = "#F4F7FB"


def setup_axes(fig, axes) -> None:
    fig.patch.set_facecolor(BG)
    for ax in np.asarray(axes).flat:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def plot_hourly_patterns(features: pd.DataFrame, target: pd.Series) -> None:
    hour = features["hour"].astype(int)
    day_type = np.where(features["working_day"].astype(str).eq("1"), "Working day", "Weekend / holiday")
    profile = pd.DataFrame({"hour": hour, "rentals": target, "day_type": day_type})
    profile = profile.groupby(["hour", "day_type"], as_index=False)["rentals"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    setup_axes(fig, [ax])
    for label, color in [("Working day", NAVY), ("Weekend / holiday", TEAL)]:
        part = profile.loc[profile["day_type"].eq(label)].sort_values("hour")
        ax.plot(part["hour"], part["rentals"], color=color, linewidth=2.8, marker="o", markersize=4, label=label)
    ax.set_title("Average rentals by hour and day type", loc="left", fontsize=17, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean rentals per hour")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(frameon=False)
    fig.text(0.08, 0.02, "Descriptive averages across the full 2011–2012 dataset.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IMAGES / "hourly_demand_patterns.png", dpi=160, facecolor=BG)
    plt.close(fig)


def plot_test_period(predictions: pd.DataFrame) -> None:
    frame = predictions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    daily = frame.set_index("datetime")[["actual", "predicted"]].resample("D").sum(min_count=1)
    observed_hours = frame.set_index("datetime")["actual"].resample("D").count()
    # Do not compare daily totals for partial days (test boundaries and daylight-saving changes).
    daily = daily.where(observed_hours.eq(24))
    smooth = daily.rolling(7, min_periods=4).mean()

    fig, ax = plt.subplots(figsize=(13, 6))
    setup_axes(fig, [ax])
    ax.plot(daily.index, daily["actual"], color=NAVY, alpha=0.2, linewidth=0.8)
    ax.plot(daily.index, daily["predicted"], color=GOLD, alpha=0.25, linewidth=0.8)
    ax.plot(smooth.index, smooth["actual"], color=NAVY, linewidth=2.4, label="Actual · 7-day average")
    ax.plot(smooth.index, smooth["predicted"], color=GOLD, linewidth=2.4, label="Predicted · 7-day average")
    ax.set_title("Daily rentals across the full held-out period", loc="left", fontsize=17, fontweight="bold", color=NAVY, pad=14)
    ax.set_ylabel("Total rentals per day")
    ax.set_xlabel("Test date")
    ax.legend(frameon=False, ncol=2)
    fig.text(0.08, 0.02, "Faint lines show complete-day totals; partial days (<24 hours) are omitted. Bold lines show rolling averages.", fontsize=9, color=MUTED)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IMAGES / "test_period_forecast.png", dpi=160, facecolor=BG)
    plt.close(fig)


def plot_error_analysis(predictions: pd.DataFrame) -> None:
    frame = predictions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["hour"] = frame["datetime"].dt.hour
    frame["absolute_error"] = (frame["actual"] - frame["predicted"]).abs()
    max_value = max(float(frame["actual"].max()), float(frame["predicted"].max())) * 1.03
    mae_by_hour = frame.groupby("hour")["absolute_error"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    setup_axes(fig, axes)
    axes[0].scatter(frame["actual"], frame["predicted"], s=10, alpha=0.18, color=TEAL, edgecolors="none")
    axes[0].plot([0, max_value], [0, max_value], linestyle="--", color=CORAL, linewidth=1.8, label="Perfect prediction")
    axes[0].set_xlim(0, max_value)
    axes[0].set_ylim(0, max_value)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Actual vs. predicted", loc="left", fontsize=15, fontweight="bold", color=NAVY, pad=12)
    axes[0].set_xlabel("Actual rentals per hour")
    axes[0].set_ylabel("Predicted rentals per hour")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(mae_by_hour.index, mae_by_hour.values, color=GOLD, width=0.78)
    axes[1].set_title("Test MAE by hour", loc="left", fontsize=15, fontweight="bold", color=NAVY, pad=12)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Mean absolute error (bikes)")
    axes[1].set_xticks(range(0, 24, 2))
    fig.suptitle("Held-out test error analysis", x=0.06, y=0.99, ha="left", fontsize=19, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(IMAGES / "test_error_analysis.png", dpi=160, facecolor=BG)
    plt.close(fig)


def plot_season_weather(features: pd.DataFrame, target: pd.Series) -> None:
    frame = pd.DataFrame({
        "season_code": features["season"].astype(int),
        "weather_code": features["weather"].astype(int),
        "rentals": target,
    })
    season_stats = frame.groupby("season_code")["rentals"].agg(["mean", "size"])
    weather_stats = frame.groupby("weather_code")["rentals"].agg(["mean", "size"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    setup_axes(fig, axes)
    axes[0].bar([f"Season {code}" for code in season_stats.index], season_stats["mean"],
                color=[TEAL, NAVY, GOLD, CORAL][:len(season_stats)])
    for index, (_, row) in enumerate(season_stats.iterrows()):
        axes[0].text(index, row["mean"] + 5, f"n={int(row['size']):,}", ha="center", fontsize=8, color=MUTED)
    axes[0].set_title("Mean demand by season code", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Mean rentals per hour")

    axes[1].bar([f"Weather {code}" for code in weather_stats.index], weather_stats["mean"],
                color=[TEAL, NAVY, GOLD, CORAL][:len(weather_stats)])
    for index, (_, row) in enumerate(weather_stats.iterrows()):
        axes[1].text(index, row["mean"] + 5, f"n={int(row['size']):,}", ha="center", fontsize=8, color=MUTED)
    axes[1].set_title("Mean demand by weather code", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylabel("Mean rentals per hour")
    fig.suptitle("Descriptive demand profiles", x=0.06, y=0.99, ha="left", fontsize=19, fontweight="bold", color=NAVY)
    fig.text(0.08, 0.02, "Unadjusted means; n above bars shows row counts. These charts do not imply causal effects.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(IMAGES / "season_weather_profiles.png", dpi=160, facecolor=BG)
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    features, target, _, _ = get_data()
    predictions_path = IMAGES.parent / "artifacts" / "test_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    plot_hourly_patterns(features, target)
    plot_test_period(predictions)
    plot_error_analysis(predictions)
    plot_season_weather(features, target)
    print(f"Saved four additional charts under {IMAGES}")


if __name__ == "__main__":
    main()
