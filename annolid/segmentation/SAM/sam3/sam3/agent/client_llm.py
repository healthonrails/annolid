# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import base64
import json
import os
from typing import Any, Optional

from openai import OpenAI


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
):
    """
    Sends a request to the OpenAI-compatible API endpoint using the OpenAI client library.

    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        max_tokens (int): Maximum number of tokens to generate (default: 4096)

    Returns:
        str: The generated response text from the server.
    """
    # Process messages to convert image paths to base64
    processed_messages = []
    for message in messages:
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

    # Create OpenAI client with custom base URL
    client = OpenAI(api_key=api_key, base_url=server_url)

    try:
        print(f"🔍 Calling model {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=processed_messages,
            max_completion_tokens=max_tokens,
            n=1,
            tools=_SAM3_TOOL_SCHEMAS,
            tool_choice="auto",
        )
        # print(f"Received response: {response.choices[0].message}")

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

    except Exception as e:
        print(f"Request failed: {e}")
        return None


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """
    Run inference on a vLLM model instance directly without using a server.

    Args:
        llm: Initialized vLLM LLM instance (passed from external initialization)
        messages: List of message dicts with role and content (OpenAI format)
        sampling_params: vLLM SamplingParams instance (initialized externally)

    Returns:
        str: Generated response text, or None if inference fails
    """
    try:
        # Process messages to handle images (convert to base64 if needed)
        processed_messages = []
        for message in messages:
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

        print("🔍 Running direct inference with vLLM...")

        # Run inference using vLLM's chat interface
        outputs = llm.chat(
            messages=processed_messages,
            sampling_params=sampling_params,
        )

        # Extract the generated text from the first output
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return generated_text
        else:
            print(f"Unexpected output format: {outputs}")
            return None

    except Exception as e:
        print(f"Direct inference failed: {e}")
        return None
