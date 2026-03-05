from __future__ import annotations

import os
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def _naive_parse_order(text: str) -> Dict[str, Any]:
    """
    Fallback parser when OpenAI is not available.
    Very simple heuristic:
    - Split text on 'and', ',', '&'
    - For each chunk, look for a leading quantity (digit or number word)
    - Remaining words form the item name.
    """
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    # Strip common polite phrases so they don't become part of item names
    cleaned_text = re.sub(
        r"\b(i want|i would like|can i get|could i get|please give me|please can i have)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    items: List[Dict[str, Any]] = []
    parts = re.split(r"\band\b|,|&", cleaned_text, flags=re.IGNORECASE)

    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        tokens = cleaned.split()
        qty = 1
        name_tokens = tokens

        if tokens:
            first = tokens[0].lower()
            if first.isdigit():
                qty = int(first)
                name_tokens = tokens[1:]
            elif first in number_words:
                qty = number_words[first]
                name_tokens = tokens[1:]

        name = " ".join(name_tokens).strip()
        if not name:
            continue
        items.append({"name": name, "qty": qty})

    if not items:
        return {"items": []}
    return {"items": items}


def extract_order_from_text(text: str) -> Dict[str, Any]:
    """
    Use GPT to extract a structured order JSON from natural language.
    Falls back to a naive rule-based parser if OpenAI is not available
    or any error occurs.
    """
    try:
        client = get_openai_client()
    except Exception:
        return _naive_parse_order(text)

    system_prompt = (
        "You are an assistant for a restaurant POS system. "
        "Given a customer's spoken order, extract a structured JSON with this shape:\n"
        '{ "items": [ { "name": "<item name>", "qty": <integer>, "size": "<optional size>" } ] }\n'
        "Only return valid JSON. Do not include explanations."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content
        if not content:
            return _naive_parse_order(text)

        import json

        data = json.loads(content)
        if "items" not in data:
            data["items"] = []
        return data
    except Exception:
        return _naive_parse_order(text)

