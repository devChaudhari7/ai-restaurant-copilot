from __future__ import annotations

import os
from typing import BinaryIO

import requests
from dotenv import load_dotenv


load_dotenv()


def _get_deepgram_api_key() -> str:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set in the environment.")
    return api_key


def transcribe_audio(file: BinaryIO, language: str | None = None) -> str:
    """
    Transcribe audio using Deepgram's /listen API.
    Supports multilingual audio based on Deepgram model configuration.
    """
    api_key = _get_deepgram_api_key()

    # Deepgram recommends streaming audio or sending raw bytes with appropriate mimetype.
    audio_bytes = file.read()

    params = {
        "model": "nova-2",
        "smart_format": "true",
    }
    # Deepgram auto-detects language for many models; language hint can be passed if desired
    if language and language != "auto":
        params["language"] = language

    response = requests.post(
        "https://api.deepgram.com/v1/listen",
        params=params,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/octet-stream",
        },
        data=audio_bytes,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Extract the transcript text from Deepgram response
    return data["results"]["channels"][0]["alternatives"][0]["transcript"]

