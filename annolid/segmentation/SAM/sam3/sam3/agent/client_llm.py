# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import base64
import json
import os
import threading
from typing import Any, Optional

from openai import OpenAI

from annolid.utils.cancellation import (
    InferenceCancelledError,
    is_stop_requested,
    torch_cancellable_forward,
)


_SAM3_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "segment_phrase",
            "description": "Ground all instances of a simple noun phrase in the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_prompt": {
                        "type": "string",
                        "description": "A short and simple noun phrase.",
                    }
                },
                "required": ["text_prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "examine_each_mask",
            "description": "Inspect each rendered mask independently.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_masks_and_return",
            "description": "Select the final answer masks from the most recent image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer_masks": {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                },
                "required": ["final_answer_masks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_no_mask",
            "description": "Report that no mask can match the prompt.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def get_image_base64_and_mime(image_path):
    """Convert image file to base64 string and get MIME type"""
    try:
        # Get MIME type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")  # Default to JPEG

        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, None


def _serialize_tool_calls(message: Any) -> Optional[str]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls", None) or []
    else:
        tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None

    serialized_calls = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            function = tool_call.get("function", None)
            call_id = str(tool_call.get("id", "") or "")
        else:
            function = getattr(tool_call, "function", None)
            call_id = str(getattr(tool_call, "id", "") or "")
        if function is None:
            continue
        if isinstance(function, dict):
            name = str(function.get("name", "") or "")
            arguments = function.get("arguments", "") or ""
        else:
            name = getattr(function, "name", "") or ""
            arguments = getattr(function, "arguments", "") or ""
        if not name:
            continue
        if isinstance(arguments, (dict, list)):
            parameters = arguments
        else:
            try:
                parameters = json.loads(arguments) if arguments else {}
            except Exception:
                parameters = arguments
        call_payload = {"name": name, "parameters": parameters}
        if call_id:
            call_payload["id"] = call_id
        serialized_calls.append(call_payload)

    if not serialized_calls:
        return None
    if len(serialized_calls) == 1:
        return f"<tool>{json.dumps(serialized_calls[0])}</tool>"
    return "".join(f"<tool>{json.dumps(call)}</tool>" for call in serialized_calls)


def send_generate_request(
    messages,
    server_url=None,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,
    max_tokens=4096,
    *,
    timeout: Optional[float] = 120.0,
    stop_event: Optional[Any] = None,
):
    """
    Sends a request to the OpenAI-compatible API endpoint using the OpenAI client library.

    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        max_tokens (int): Maximum number of tokens to generate (default: 4096)
        timeout (float, optional): Bounded HTTP socket/request timeout in seconds.
        stop_event (Event, optional): Cooperative cancellation event; closes active HTTP sockets on set().

    Returns:
        str: The generated response text from the server.
    """
    if is_stop_requested(stop_event):
        raise InferenceCancelledError("SAM3 agent LLM request cancelled.")

    # Process messages to convert image paths to base64
    processed_messages = []
    for message in messages:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM3 agent LLM request cancelled.")
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    # Convert image path to base64 format
                    image_path = c["image"]

                    print("image_path", image_path)
                    new_image_path = image_path.replace(
                        "?", "%3F"
                    )  # Escape ? in the path

                    # Read the image file and convert to base64
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(
                            new_image_path
                        )
                        if base64_image is None:
                            print(
                                f"Warning: Could not convert image to base64: {new_image_path}"
                            )
                            continue

                        # Create the proper image_url structure with base64 data
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                        )

                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {new_image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {new_image_path}: {e}")
                        continue
                else:
                    processed_content.append(c)

            processed_message["content"] = processed_content
        processed_messages.append(processed_message)

    if is_stop_requested(stop_event):
        raise InferenceCancelledError("SAM3 agent LLM request cancelled.")

    # Create OpenAI client with custom base URL and bounded socket timeout
    try:
        client = OpenAI(api_key=api_key, base_url=server_url, timeout=timeout)
    except TypeError:
        client = OpenAI(api_key=api_key, base_url=server_url)

    def _close_client() -> None:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    unsubscribe = None
    watcher_done: Optional[threading.Event] = None
    if stop_event is not None:
        add_listener = getattr(stop_event, "add_listener", None)
        if callable(add_listener):
            unsubscribe = add_listener(_close_client)
        elif callable(getattr(stop_event, "wait", None)):
            watcher_done = threading.Event()

            def _poll_stop() -> None:
                while not watcher_done.is_set():
                    if is_stop_requested(stop_event):
                        _close_client()
                        return
                    watcher_done.wait(0.05)

            threading.Thread(
                target=_poll_stop,
                name="sam3-agent-llm-cancel",
                daemon=True,
            ).start()

    try:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM3 agent LLM request cancelled.")
        print(f"🔍 Calling model {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=processed_messages,
            max_completion_tokens=max_tokens,
            n=1,
            tools=_SAM3_TOOL_SCHEMAS,
            tool_choice="auto",
        )
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM3 agent LLM request cancelled.")

        # Extract the response content
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content", None)
            if content is not None and str(content).strip():
                return content
            tool_call_text = _serialize_tool_calls(message)
            if tool_call_text is not None:
                return tool_call_text
            print(
                "Unexpected response format: assistant message had no text content or tool calls."
            )
            return None
        else:
            print(f"Unexpected response format: {response}")
            return None

    except InferenceCancelledError:
        raise
    except Exception as e:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM3 agent LLM request cancelled.") from e
        print(f"Request failed: {e}")
        return None
    finally:
        if watcher_done is not None:
            watcher_done.set()
        if callable(unsubscribe):
            try:
                unsubscribe()
            except Exception:
                pass
        _close_client()


def build_hf_stopping_criteria(stop_event: Optional[Any]) -> Optional[Any]:
    """Build a token-level ``StoppingCriteriaList`` wired to ``stop_event``.

    Imports ``transformers`` lazily so GUI/CLI startup never depends on it.
    Falls back to a callable criterion list when ``transformers`` is absent.
    """
    if stop_event is None:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StopEventCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                _ = input_ids, scores, kwargs
                return is_stop_requested(stop_event)

        return StoppingCriteriaList([_StopEventCriteria()])
    except Exception:

        class _FallbackStopCriteria:
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                _ = input_ids, scores, kwargs
                return is_stop_requested(stop_event)

        return [_FallbackStopCriteria()]


def _extract_pipeline_generated_text(outputs: Any) -> Optional[str]:
    if outputs is None:
        return None
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict) and "generated_text" in first:
            gen = first["generated_text"]
            if isinstance(gen, str):
                return gen
            if isinstance(gen, list) and gen:
                last_msg = gen[-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = [
                            str(item.get("text", ""))
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        return "".join(parts) if parts else None
        if isinstance(first, str):
            return first
    return None


def send_local_pipeline_request(
    pipe: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 4096,
    *,
    stop_event: Optional[Any] = None,
    **generate_kwargs: Any,
) -> Optional[str]:
    """Run local HuggingFace ``transformers.pipeline`` VLM generation with token-level cancellation."""
    if is_stop_requested(stop_event):
        raise InferenceCancelledError("SAM3 agent local pipeline generation cancelled.")

    stopping_criteria = build_hf_stopping_criteria(stop_event)
    call_kwargs = dict(generate_kwargs)
    call_kwargs.setdefault("max_new_tokens", max_new_tokens)
    if stopping_criteria is not None:
        existing = call_kwargs.get("stopping_criteria")
        if existing is None:
            call_kwargs["stopping_criteria"] = stopping_criteria
        elif isinstance(existing, list):
            call_kwargs["stopping_criteria"] = list(existing) + list(stopping_criteria)

    model = getattr(pipe, "model", None)
    try:
        with torch_cancellable_forward(model, stop_event):
            try:
                outputs = pipe(messages, **call_kwargs)
            except TypeError as exc:
                if (
                    "stopping_criteria" not in str(exc)
                    or "stopping_criteria" not in call_kwargs
                ):
                    raise
                fallback_kwargs = dict(call_kwargs)
                fallback_kwargs.pop("stopping_criteria", None)
                outputs = pipe(messages, **fallback_kwargs)
    except InferenceCancelledError:
        raise
    except Exception as exc:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError(
                "SAM3 agent local pipeline generation cancelled."
            ) from exc
        print(f"Local pipeline inference failed: {exc}")
        return None

    if is_stop_requested(stop_event):
        raise InferenceCancelledError("SAM3 agent local pipeline generation cancelled.")
    return _extract_pipeline_generated_text(outputs)


send_generate_request_orig = send_local_pipeline_request


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any = None,
    *,
    stop_event: Optional[Any] = None,
) -> Optional[str]:
    """
    Run inference on a vLLM model instance or local HF pipeline directly without using a server.

    Args:
        llm: Initialized vLLM LLM instance or HuggingFace pipeline callable
        messages: List of message dicts with role and content (OpenAI format)
        sampling_params: vLLM SamplingParams instance (optional for HF pipelines)
        stop_event: Optional cooperative cancellation event

    Returns:
        str: Generated response text, or None if inference fails
    """
    if is_stop_requested(stop_event):
        raise InferenceCancelledError("SAM3 agent direct LLM inference cancelled.")
    if not hasattr(llm, "chat") and callable(llm):
        max_tokens = int(getattr(sampling_params, "max_tokens", 4096) or 4096)
        return send_local_pipeline_request(
            llm,
            messages,
            max_new_tokens=max_tokens,
            stop_event=stop_event,
        )

    try:
        # Process messages to handle images (convert to base64 if needed)
        processed_messages = []
        for message in messages:
            if is_stop_requested(stop_event):
                raise InferenceCancelledError(
                    "SAM3 agent direct LLM inference cancelled."
                )
            processed_message = message.copy()
            if message["role"] == "user" and "content" in message:
                processed_content = []
                for c in message["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        # Convert image path to base64 format
                        image_path = c["image"]
                        new_image_path = image_path.replace("?", "%3F")

                        try:
                            base64_image, mime_type = get_image_base64_and_mime(
                                new_image_path
                            )
                            if base64_image is None:
                                print(
                                    f"Warning: Could not convert image: {new_image_path}"
                                )
                                continue

                            # vLLM expects image_url format
                            processed_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                }
                            )
                        except Exception as e:
                            print(
                                f"Warning: Error processing image {new_image_path}: {e}"
                            )
                            continue
                    else:
                        processed_content.append(c)

                processed_message["content"] = processed_content
            processed_messages.append(processed_message)

        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM3 agent direct LLM inference cancelled.")

        print("🔍 Running direct inference with vLLM...")

        # Run inference using vLLM's chat interface
        outputs = llm.chat(
            messages=processed_messages,
            sampling_params=sampling_params,
        )
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM3 agent direct LLM inference cancelled.")

        # Extract the generated text from the first output
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return generated_text
        else:
            print(f"Unexpected output format: {outputs}")
            return None

    except InferenceCancelledError:
        raise
    except Exception as e:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError(
                "SAM3 agent direct LLM inference cancelled."
            ) from e
        print(f"Direct inference failed: {e}")
        return None
