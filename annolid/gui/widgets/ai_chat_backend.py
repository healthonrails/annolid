from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional

from qtpy import QtCore
from qtpy.QtCore import QMetaObject, QRunnable


class StreamingChatTask(QRunnable):
    """Stream a chat response from the selected provider back to a widget."""

    def __init__(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        widget=None,
        model: str = "llama3.2-vision:latest",
        provider: str = "ollama",
        settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.prompt = prompt
        self.image_path = image_path
        self.widget = widget
        self.model = model
        self.provider = provider
        self.settings = settings or {}

    def run(self) -> None:
        try:
            if self.provider == "ollama":
                self._run_ollama()
            elif self.provider == "openai":
                self._run_openai()
            elif self.provider == "gemini":
                self._run_gemini()
            else:
                raise ValueError(f"Unsupported provider '{self.provider}'.")
        except Exception as exc:
            self._emit_final(f"Error in chat interaction: {exc}", is_error=True)

    def _emit_chunk(self, chunk: str) -> None:
        QMetaObject.invokeMethod(
            self.widget,
            "stream_chat_chunk",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, chunk),
        )

    def _emit_final(self, message: str, *, is_error: bool) -> None:
        QMetaObject.invokeMethod(
            self.widget,
            "update_chat_response",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, message),
            QtCore.Q_ARG(bool, is_error),
        )

    def _run_ollama(self) -> None:
        ollama_module = globals().get("ollama")
        if ollama_module is None:
            try:
                ollama_module = importlib.import_module("ollama")
            except ImportError as exc:
                raise ImportError(
                    "The python 'ollama' package is not installed."
                ) from exc
            globals()["ollama"] = ollama_module

        host = self.settings.get("ollama", {}).get("host")
        prev_host_present = "OLLAMA_HOST" in os.environ
        prev_host_value = os.environ.get("OLLAMA_HOST")
        try:
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)

            messages = [{"role": "user", "content": self.prompt}]
            if self.image_path and os.path.exists(self.image_path):
                messages[0]["images"] = [self.image_path]

            stream = ollama_module.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )
            full_response = ""
            for part in stream:
                if "message" in part and "content" in part["message"]:
                    chunk = part["message"]["content"]
                    full_response += chunk
                    self._emit_chunk(chunk)
                elif "error" in part:
                    self._emit_final(f"Stream error: {part['error']}", is_error=True)
                    return

            if not full_response.strip():
                self._emit_final("No response from Ollama.", is_error=True)
            else:
                self._emit_final("", is_error=False)
        finally:
            if prev_host_present and prev_host_value is not None:
                os.environ["OLLAMA_HOST"] = prev_host_value
            else:
                os.environ.pop("OLLAMA_HOST", None)

    def _run_openai(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for GPT providers."
            ) from exc

        config = self.settings.get("openai", {})
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        if not api_key:
            raise ValueError("OpenAI API key is missing. Configure it in settings.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        user_prompt = self.prompt
        if self.image_path and os.path.exists(self.image_path):
            user_prompt += (
                f"\n\n[Note: Image context available at {self.image_path}. "
                "Use this visual context in your response.]"
            )

        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        model_lower = (self.model or "").lower()
        if "gpt-5" not in model_lower:
            request_payload["temperature"] = 0.7

        response = client.chat.completions.create(**request_payload)
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""
        self._emit_final(text, is_error=False)

    def _run_gemini(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for Gemini providers."
            ) from exc

        config = self.settings.get("gemini", {})
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key is missing. Configure it in settings.")

        genai.configure(api_key=api_key)
        model_name = self.model or "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)

        user_prompt = self.prompt
        if self.image_path and os.path.exists(self.image_path):
            user_prompt += (
                f"\n\n[Note: Image context available at {self.image_path}. "
                "Use this visual context in your response.]"
            )

        result = model.generate_content(user_prompt)
        text = getattr(result, "text", "") or ""
        self._emit_final(text, is_error=False)
