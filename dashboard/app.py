from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path so `ai_modules` and `backend` are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_modules.whisper_client import transcribe_audio


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def upload_pos_csv_to_backend(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{BACKEND_URL}/upload-pos-data", files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_menu_insights() -> Dict[str, Any]:
    resp = requests.get(f"{BACKEND_URL}/menu-insights", timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_combo_recommendations() -> Dict[str, Any]:
    resp = requests.get(f"{BACKEND_URL}/combo-recommendations", timeout=60)
    resp.raise_for_status()
    return resp.json()


def send_voice_order(text: str) -> Dict[str, Any]:
    resp = requests.post(f"{BACKEND_URL}/voice-order", json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def create_order(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    resp = requests.post(f"{BACKEND_URL}/create-order", json={"items": items}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def page_menu_insights() -> None:
    st.header("📊 POS Analytics & Menu Engineering")

    st.subheader("Upload POS CSV")
    uploaded_file = st.file_uploader("Upload POS data (CSV)", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        use_sample = st.button("Use Sample Dataset")
    with col2:
        backend_status = st.empty()

    if uploaded_file is not None:
        try:
            backend_status.info("Uploading to backend...")
            data = upload_pos_csv_to_backend(uploaded_file.read(), uploaded_file.name)
            backend_status.success(f"Uploaded {data['rows']} rows.")
            st.session_state["last_upload"] = data
        except Exception as e:
            backend_status.error(f"Upload failed: {e}")

    if use_sample:
        # Look for sample data either in project root or in data/ folder
        project_root = PROJECT_ROOT
        candidate_paths = [
            project_root / "sample_pos_data.csv",
            project_root / "data" / "sample_pos_data.csv",
        ]
        sample_path = next((p for p in candidate_paths if p.exists()), None)

        if sample_path and sample_path.exists():
            with open(sample_path, "rb") as f:
                try:
                    backend_status.info(f"Uploading sample dataset from {sample_path} to backend...")
                    data = upload_pos_csv_to_backend(f.read(), "sample_pos_data.csv")
                    backend_status.success(f"Uploaded {data['rows']} rows from sample dataset.")
                    st.session_state["last_upload"] = data
                except Exception as e:
                    backend_status.error(f"Sample upload failed: {e}")
        else:
            backend_status.error("Sample dataset not found. Please generate it with `python data/generate_sample_pos_data.py`.")

    st.markdown("---")
    st.subheader("Menu Performance")

    try:
        insights_payload = fetch_menu_insights()
        items = insights_payload.get("menu_insights", [])
        if items:
            df = pd.DataFrame(items)
            st.dataframe(df)

            st.subheader("Contribution Margin by Item")
            if "total_contribution_margin" in df.columns:
                st.bar_chart(df.set_index("item_name")["total_contribution_margin"])

            st.subheader("Item Popularity (Quantity Sold)")
            if "quantity_sold" in df.columns:
                st.bar_chart(df.set_index("item_name")["quantity_sold"])
        else:
            st.info("Upload POS data to see menu insights.")
    except Exception as e:
        st.warning(f"Unable to load menu insights: {e}")

    st.markdown("---")
    st.subheader("Combo Recommendations")
    try:
        combos_payload = fetch_combo_recommendations()
        combos = combos_payload.get("combos", [])
        if combos:
            st.table(pd.DataFrame(combos))
        else:
            st.info("No combo recommendations available. Try with more POS data.")
    except Exception as e:
        st.warning(f"Unable to load combo recommendations: {e}")


def page_voice_order() -> None:
    st.header("🎙️ Voice Ordering Copilot")

    st.markdown("Record or upload audio, then let the AI parse the order, suggest upsells, and generate a summary.")

    st.subheader("1. Record or Upload Audio")
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
    language = st.selectbox("Language (hint for transcription)", ["auto", "en", "hi"], index=0)

    transcript_text = st.text_area("Or enter order text directly", height=100)

    if "transcript" not in st.session_state:
        st.session_state["transcript"] = ""

    if st.button("Transcribe Audio with Whisper") and audio_file is not None:
        try:
            st.info("Transcribing audio...")
            language_arg = None if language == "auto" else language
            # Convert uploaded file to a BytesIO for Whisper client
            audio_bytes = io.BytesIO(audio_file.read())
            transcript = transcribe_audio(audio_bytes, language=language_arg)
            st.session_state["transcript"] = transcript
            st.success("Transcription complete.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")

    st.subheader("2. Transcription")
    if st.session_state.get("transcript"):
        st.write(st.session_state["transcript"])
    elif transcript_text:
        st.write(transcript_text)
    else:
        st.info("Provide audio or text to continue.")

    st.subheader("3. Parse Order and Upsell Suggestions")
    if st.button("Parse Order with GPT"):
        text_source = st.session_state.get("transcript") or transcript_text
        if not text_source:
            st.warning("Please provide audio or text first.")
        else:
            try:
                parsed = send_voice_order(text_source)
                st.session_state["parsed_order"] = parsed
                st.success("Order parsed successfully.")
            except Exception as e:
                st.error(f"Failed to parse order: {e}")

    parsed_order = st.session_state.get("parsed_order")
    if parsed_order:
        st.json(parsed_order)

        st.subheader("4. Confirm and Create Order")
        items = parsed_order.get("items", [])
        editable_items: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                name = st.text_input(f"Item {idx+1} Name", value=item.get("name", ""))
            with col2:
                qty = st.number_input(f"Qty {idx+1}", min_value=1, value=int(item.get("qty", 1)))
            with col3:
                size = st.text_input(f"Size {idx+1}", value=item.get("size") or "")
            editable_items.append({"name": name, "qty": qty, "size": size or None})

        if st.button("Create Order"):
            try:
                order_resp = create_order(editable_items)
                st.success("Order created (simulated)!")
                st.subheader("Order Summary")
                st.json(order_resp)
            except Exception as e:
                st.error(f"Failed to create order: {e}")


def main() -> None:
    st.set_page_config(page_title="AI-Powered Restaurant Copilot", layout="wide")
    st.sidebar.title("AI Revenue & Voice Copilot")
    page = st.sidebar.radio("Navigate", ["Menu Insights", "Voice Ordering"])

    if page == "Menu Insights":
        page_menu_insights()
    else:
        page_voice_order()


if __name__ == "__main__":
    main()

