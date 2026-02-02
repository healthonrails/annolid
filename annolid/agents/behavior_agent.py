import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from PIL import Image

from annolid.core.media.video import CV2Video
from annolid.core.models.adapters.llm_chat import LLMChatAdapter
from annolid.core.models.base import ModelRequest
from annolid.utils.llm_settings import (
    ensure_provider_env,
    resolve_llm_config,
)

PROFILE_NAME = "behavior_agent"
SYSTEM_PROMPT = """
Task: Video Behavior Analysis

You will be provided with a video depicting mouse behavior. Your goal is to describe the video and classify the behaviors observed with sufficient detail to ensure accurate behavior identification. Use the behavior list provided below to identify and rank the top 5 behaviors most confidently observed.

Behavior List
    •   A. Approach
    •   B. Nose-to-Nose Sniffing
    •   C. Stimulus Mouse Sniffing FP Mouse
    •   D. Nose-to-Flank Sniffing
    •   E. Nose-to-Anogenital Sniffing
    •   F. Withdrawal
    •   G. FP Mouse Sniffing Excretions
    •   H. Rearing
    •   I. Grooming
    •   J. FP Mouse Mounting Stimulus Mouse
    •   K. FP Mouse Snigging Excretions
    •   L. FP Mouse Tail Rattling
    •   M. Others

Instructions
    1.  Analyze the Video:
Examine the video to identify the most dominant and relevant behaviors.
    2.  Rank Behaviors:
Rank the top 5 observed behaviors by confidence (most likely) and relevance (most descriptive). Use the behavior list above.
    3.  Justify Observations (Optional):
If clarity is required, provide a 1-2 sentence explanation for each selected behavior to support your ranking.

Output Format

1. <Behavior Code> (<Behavior Description>): <Optional Justification>
2. <Behavior Code> (<Behavior Description>): <Optional Justification>
...

Example Output

1. B (Nose-to-Nose Sniffing): Direct nose-to-nose interaction is visible.
2. A (Approach): The FP mouse approaches the stimulus mouse.
3. D (Nose-to-Flank Sniffing): Flank sniffing observed.
4. F (Withdrawal): The FP mouse retreats after interaction.
5. M (Others): General movement across the enclosure.

...

Important Notes for Classification
    •   Use context clues (e.g., proximity, positioning, and movement) to distinguish between similar behaviors.
    •   Pay close attention to mice body parts like nose, tailbase, paws and other moving body parts.
    •   If multiple behaviors occur simultaneously, prioritize based on relevance to overall interaction.
"""

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class AgentRunResult:
    """Simple response wrapper aligned with the previous `result.content` usage."""

    content: str


class BehaviorVideoAgent:
    """Provider-configured behavior agent for video analysis."""

    def __init__(self, provider: str, model_name: str, system_prompt: str) -> None:
        self._provider = str(provider).strip().lower()
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._model = None
        self._chat_adapter = None

        if self._provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "The 'google-generativeai' package is required for Gemini providers."
                ) from exc

            self._model = genai.GenerativeModel(
                model_name=model_name, system_instruction=system_prompt
            )
        else:
            self._chat_adapter = LLMChatAdapter(
                profile=PROFILE_NAME,
                provider=self._provider,
                model=self._model_name,
            )

    def run(self, prompt: str, videos: Sequence[Any]) -> AgentRunResult:
        if not videos:
            raise ValueError("At least one uploaded video is required.")
        if self._model is None:
            raise RuntimeError("BehaviorVideoAgent is not initialized.")

        response = self._model.generate_content([prompt, *videos])
        text = getattr(response, "text", "") or ""
        return AgentRunResult(content=text)

    def _sample_frame_indices(self, total_frames: int, max_frames: int) -> list[int]:
        if total_frames <= 0:
            return []
        if total_frames <= max_frames:
            return list(range(total_frames))
        if max_frames == 1:
            return [0]

        step = (total_frames - 1) / float(max_frames - 1)
        indices = {int(round(i * step)) for i in range(max_frames)}
        ordered = sorted(indices)
        cursor = 0
        while len(ordered) < max_frames and cursor < total_frames:
            if cursor not in indices:
                ordered.append(cursor)
            cursor += 1
        ordered.sort()
        return ordered[:max_frames]

    def run_from_video_path(self, video_path: str, user_prompt: str) -> AgentRunResult:
        if self._provider == "gemini":
            try:
                from google.generativeai import get_file, upload_file  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "The 'google-generativeai' package is required for Gemini video analysis."
                ) from exc

            print("Uploading video...")
            video_file = upload_file(video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = get_file(video_file.name)

            prompt = f"""
            First analyze this video and then answer the following question using
            the video analysis: {user_prompt}

            Provide a comprehensive response focusing on practical, actionable information.
            """
            print("Analyzing video and performing research...")
            return self.run(prompt, videos=[video_file])

        if self._chat_adapter is None:
            raise RuntimeError("BehaviorVideoAgent is not initialized.")

        video = CV2Video(video_path)
        frame_notes: list[str] = []
        try:
            fps = max(1.0, float(video.fps() or 0.0))
            frame_indices = self._sample_frame_indices(
                video.total_frames(), max_frames=8
            )
            caption_prompt = "Describe mouse positions, interactions, and behavior cues in this frame."

            with TemporaryDirectory(prefix="annolid_behavior_frames_") as tmp_dir:
                for idx in frame_indices:
                    try:
                        frame_rgb = video.load_frame(idx)
                    except Exception:
                        continue

                    frame_path = Path(tmp_dir) / f"frame_{idx:08d}.png"
                    Image.fromarray(frame_rgb).save(frame_path)
                    ts = idx / fps if fps > 0 else 0.0

                    try:
                        response = self._chat_adapter.predict(
                            ModelRequest(
                                task="caption",
                                text=caption_prompt,
                                image_path=str(frame_path),
                                params={"temperature": 0.1, "max_tokens": 180},
                            )
                        )
                        caption_text = (response.text or "").strip()
                    except Exception as exc:
                        caption_text = f"(caption failed: {exc})"

                    frame_notes.append(f"- t={ts:.2f}s frame={idx}: {caption_text}")
        finally:
            video.release()

        if not frame_notes:
            raise RuntimeError("Unable to extract usable frames for behavior analysis.")

        final_prompt = (
            "You are given frame-level observations sampled in chronological order.\n"
            "Use them to infer likely behaviors over the full video and answer the request.\n\n"
            f"User request: {user_prompt}\n\n"
            "Frame observations:\n" + "\n".join(frame_notes)
        )

        response = self._chat_adapter.predict(
            ModelRequest(
                task="chat",
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": final_prompt},
                ],
                params={"temperature": 0.1},
            )
        )
        return AgentRunResult(content=(response.text or "").strip())


def initialize_agent() -> BehaviorVideoAgent:
    """Initialize and return the configured video behavior analysis agent."""
    config = resolve_llm_config(profile=PROFILE_NAME)
    ensure_provider_env(config)

    if config.provider == "gemini":
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for Gemini providers."
            ) from exc
        genai.configure(api_key=config.api_key)

    return BehaviorVideoAgent(
        provider=config.provider,
        model_name=config.model,
        system_prompt=SYSTEM_PROMPT,
    )


def process_video_with_agent(video_path, user_prompt, agent):
    """
    Analyze a video and perform research based on the user's prompt.

    Args:
        video_path (str): Path to the video file.
        user_prompt (str): The question or prompt for the agent.
        agent (BehaviorVideoAgent): The initialized video behavior agent.

    Returns:
        str: The response from the agent.
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        result = agent.run_from_video_path(
            video_path=video_path, user_prompt=user_prompt
        )
        return result.content

    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    # Example usage
    # Replace with your video file path
    video_path = os.path.expanduser("Stimulus Mouse Sniffing FP Mouse_3.16-7.64.mp4")
    user_prompt = (
        """Describe the main activities and classify behavior in this video."""
    )

    # Initialize the agent
    agent = initialize_agent()

    # Process the video and get the result
    response = process_video_with_agent(video_path, user_prompt, agent)
    print("Agent Response:")
    print(response)
