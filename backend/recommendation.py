from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def _build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-hot encoded basket matrix for market basket analysis.
    """
    basket = (
        df.groupby(["order_id", "item_name"])["quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )
    # Convert quantities to 0/1
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket


def get_top_combos(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    basket = _build_basket_matrix(df)
    if basket.empty:
        return []

    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    if frequent_itemsets.empty:
        return []

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    if rules.empty:
        return []

    # Focus on pairs for simplicity
    rules = rules[rules["antecedents"].apply(len) == 1]
    rules = rules[rules["consequents"].apply(len) == 1]
    rules = rules.sort_values(["confidence", "lift"], ascending=False).head(top_n)

    combos: List[Dict[str, Any]] = []
    for _, row in rules.iterrows():
        antecedent = list(row["antecedents"])[0]
        consequent = list(row["consequents"])[0]
        combos.append(
            {
                "combo": f"{antecedent} + {consequent}",
                "antecedent": antecedent,
                "consequent": consequent,
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            }
        )
    return combos


def get_upsell_suggestions(ordered_items: List[str], df: pd.DataFrame, top_n: int = 3) -> List[str]:
    """
    For any of the ordered items, suggest consequents from association rules.
    """
    basket = _build_basket_matrix(df)
    if basket.empty:
        return []

    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    if frequent_itemsets.empty:
        return []

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    if rules.empty:
        return []

    suggestions: List[str] = []
    for _, row in rules.iterrows():
        antecedent_items = list(row["antecedents"])
        consequent_items = list(row["consequents"])
        if len(antecedent_items) == 1 and len(consequent_items) == 1:
            antecedent = antecedent_items[0]
            consequent = consequent_items[0]
            if antecedent in ordered_items and consequent not in ordered_items:
                suggestions.append(consequent)

    # Deduplicate while preserving order
    deduped: List[str] = []
    for s in suggestions:
        if s not in deduped:
            deduped.append(s)

    return deduped[:top_n]

