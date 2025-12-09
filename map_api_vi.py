"""
map_api_vi.py

Minimal MapAPI wrapper for the Viavi Streamlit app.

- Default backend: OpenAI Chat API
- Interface expected by the app:
    * MapAPI()
    * generate_response(messages: list[{"role": ..., "content": ...}]) -> str

No TinySA dependencies. SLM mode in the UI currently reuses the same
OpenAI backend but can be extended later to use a local model.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None
    _HAS_OPENAI = False


class MapAPI:
    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        """
        Initialize MapAPI.

        Since your current app does not pass a local tokenizer/model,
        this implementation uses only the OpenAI backend.

        Args:
            openai_model: Name of the OpenAI chat model to use.
            temperature: Sampling temperature.
            max_tokens: Max tokens for each response.
        """
        if not _HAS_OPENAI:
            raise RuntimeError(
                "OpenAI client is not available. Install `openai` "
                "and set OPENAI_API_KEY in your environment."
            )

        # Load environment variables
        load_dotenv()

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_model = openai_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Core method used by your Streamlit app
    # ------------------------------------------------------------------
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a chat response using OpenAI.

        `messages` is a list of dicts with "role" and "content":
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          ...
        ]

        Returns:
            Assistant text response (str).
        """
        completion = self.client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        reply = completion.choices[0].message.content or ""
        return reply.strip()
