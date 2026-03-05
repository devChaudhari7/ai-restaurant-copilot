from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd


def compute_menu_engineering_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute revenue, cost, contribution margin, popularity, and menu engineering category.
    Expects columns:
    - item_name
    - quantity
    - unit_price
    - food_cost_per_unit
    """
    required_cols = {"item_name", "quantity", "unit_price", "food_cost_per_unit"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data["revenue"] = data["quantity"] * data["unit_price"]
    data["cost"] = data["quantity"] * data["food_cost_per_unit"]
    data["contribution_margin_per_unit"] = data["unit_price"] - data["food_cost_per_unit"]
    data["total_contribution_margin"] = data["quantity"] * data["contribution_margin_per_unit"]

    grouped = (
        data.groupby("item_name")
        .agg(
            quantity_sold=("quantity", "sum"),
            revenue=("revenue", "sum"),
            cost=("cost", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_food_cost=("food_cost_per_unit", "mean"),
            total_contribution_margin=("total_contribution_margin", "sum"),
            contribution_margin_per_unit=("contribution_margin_per_unit", "mean"),
        )
        .reset_index()
    )

    # Popularity: quantity_sold
    popularity_median = grouped["quantity_sold"].median()
    profit_median = grouped["contribution_margin_per_unit"].median()

    def classify_row(row: pd.Series) -> str:
        high_pop = row["quantity_sold"] >= popularity_median
        high_profit = row["contribution_margin_per_unit"] >= profit_median
        if high_pop and high_profit:
            return "Star"
        if (not high_pop) and high_profit:
            return "Puzzle"
        if high_pop and (not high_profit):
            return "Plowhorse"
        return "Dog"

    grouped["menu_category"] = grouped.apply(classify_row, axis=1)
    return grouped


def get_menu_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    metrics = compute_menu_engineering_metrics(df)
    return metrics.to_dict(orient="records")


def get_price_optimization_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    metrics = compute_menu_engineering_metrics(df)
    popularity_median = metrics["quantity_sold"].median()
    profit_median = metrics["contribution_margin_per_unit"].median()

    suggestions: List[Dict[str, Any]] = []
    for _, row in metrics.iterrows():
        if row["quantity_sold"] >= popularity_median and row["contribution_margin_per_unit"] < profit_median:
            suggestions.append(
                {
                    "item_name": row["item_name"],
                    "reason": "High popularity but low margin",
                    "suggested_action": "Consider increasing price or reducing cost",
                }
            )
    return suggestions


def get_sales_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    # Top / least selling items by quantity
    qty_by_item = df.groupby("item_name")["quantity"].sum().sort_values(ascending=False)
    result["top_selling_items"] = qty_by_item.head(5).to_dict()
    result["least_selling_items"] = qty_by_item.tail(5).to_dict()

    # Peak order hours
    if "transaction_date" in df.columns:
        ts = pd.to_datetime(df["transaction_date"])
        hours = ts.dt.hour
        hour_counts = df.copy()
        hour_counts["hour"] = hours
        peak_hours = (
            hour_counts.groupby("hour")["order_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )
        result["peak_hours"] = peak_hours

    # Busiest days of week
    if "day_of_week" in df.columns:
        day_counts = df.groupby("day_of_week")["order_id"].nunique().sort_values(ascending=False).to_dict()
        result["busiest_days"] = day_counts

    # Simple natural-language insights
    insights: List[str] = []

    if "day_of_week" in df.columns:
        item_day = (
            df.groupby(["item_name", "day_of_week"])["quantity"]
            .sum()
            .reset_index()
        )
        for item in item_day["item_name"].unique():
            subset = item_day[item_day["item_name"] == item]
            top_row = subset.sort_values("quantity", ascending=False).iloc[0]
            insights.append(f"{item} sells most on {top_row['day_of_week']}.")

    if "transaction_date" in df.columns:
        df_hours = df.copy()
        df_hours["hour"] = pd.to_datetime(df_hours["transaction_date"]).dt.hour
        bev_mask = df_hours.get("category", pd.Series(index=df_hours.index, dtype=str)).str.contains(
            "beverage", case=False, na=False
        )
        beverages = df_hours[bev_mask]
        if not beverages.empty:
            beverages["hour_bucket"] = pd.cut(
                beverages["hour"],
                bins=[0, 6, 12, 15, 18, 21, 24],
                labels=["12 AM-6 AM", "6 AM-12 PM", "12 PM-3 PM", "3 PM-6 PM", "6 PM-9 PM", "9 PM-12 AM"],
                right=False,
            )
            bucket_counts = (
                beverages.groupby("hour_bucket")["order_id"]
                .nunique()
                .sort_values(ascending=False)
            )
            if not bucket_counts.empty:
                top_bucket = bucket_counts.index[0]
                insights.append(f"Beverages sell most between {top_bucket}.")

    result["insights"] = insights
    return result

