from __future__ import annotations

from typing import Dict, Any, List

from rapidfuzz import process

from ai_modules.gpt_client import extract_order_from_text


def _fuzzy_match_item(spoken_name: str, menu_items: List[str]) -> str:
    if not menu_items:
        return spoken_name
    match, score, _ = process.extractOne(spoken_name, menu_items)
    # Simple threshold to avoid terrible matches
    if score < 60:
        return spoken_name
    return match


def normalize_items_with_menu(order: Dict[str, Any], menu_items: List[str]) -> Dict[str, Any]:
    normalized_items: List[Dict[str, Any]] = []
    for item in order.get("items", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        matched_name = _fuzzy_match_item(name, menu_items)
        normalized_items.append(
            {
                "name": matched_name,
                "qty": int(item.get("qty", 1)),
                "size": item.get("size"),
            }
        )
    return {"items": normalized_items}


def parse_order_text(text: str, menu_items: List[str]) -> Dict[str, Any]:
    """
    Use GPT to parse order text and then fuzzy match items to known menu items.
    """
    raw_order = extract_order_from_text(text)
    normalized = normalize_items_with_menu(raw_order, menu_items)
    return normalized

