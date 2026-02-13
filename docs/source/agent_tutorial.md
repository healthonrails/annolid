# Annolid Agent Tutorial

## 1. What the Agent does

- Performs detection, tracking, and behavior event proposals per frame.
- Streams NDJSON records (`agent.ndjson`) and LabelMe JSONs with enriched `otherData`.
- Saves everything in a video-specific results folder with caches and artifacts.

## 2. Running the Agent (GUI)

1. Open a video and choose **Analysis → Run Agent**.
2. Configure the behavior spec, vision/LLM adapters, sampling (stride/max frames), and optionally enable **Rerun from anchors** (clears future frames after your manual corrections).
3. Click **OK** to launch; progress appears in a modal dialog, and the GUI tails the NDJSON while it writes.

## 3. Agent Output in the GUI

- Agent-suggested events appear in the **Behavior Log**, complete with status (Confirmed/Auto).
- Timeline marks show start/end points; colors derive from schema categories.
- Unconfirmed events are excluded from exports/summaries until you confirm them.

## 4. Reviewing & Editing Events

- Double-click a log entry to view details; the context menu lets you *Edit Interval*, *Delete*, or *Confirm/Reject*.
- Editing adjusts the timeline + updates the underlying annotation store via `BehaviorController`.
- Confirmed events are marked `confirmed=true` and used in CSV exports/time budgets.

## 5. Embedding Search

- The **Embedding Search** dock (right side) lets you:
  - Search for frames similar to the current frame via Qwen3 embeddings.
  - Double-click results to jump to frames.
  - Label selected search hits as a behavior to propagate agent insights quickly.

## 6. Human-in-the-Loop Refinement

- Manual corrections become **anchors**.
- When rerunning with “Rerun from anchors,” frames after the last anchor are cleared, and the agent recomputes from that point.

## 7. Agent Mode vs Classic Mode

- Agent Mode (default) shows the behavior dock, embedding search, and agent run toolbar.
- Toggle it via **View → Agent Mode** to hide agent-specific panels for a classic labeling workflow.

## 8. Exporting Results

- Behavior exports and summaries automatically exclude auto (unconfirmed) events.
- Use **Save CSV** or **Behavior Time Budget** to generate reports from confirmed data.

## 9. Describe Behavior (Caption dock)

The **Caption** dock includes a **Describe behavior** action that can turn frames into a natural-language behavior log.

- **Segment summary (sample frames)**: pick a time range and choose how many frames to sample uniformly.
- **Timeline (describe every N seconds)**: generate timestamped lines over long videos (e.g., every `1s`), batching multiple frames per model call.
- Timeline results are persisted per timestamp in a sidecar (`<video>.<ext>.behavior_timeline.json`) and auto-loaded, so scrubbing frames inside that interval restores the saved behavior text.
- Re-running timeline mode skips already described timestamps and resumes from where the previous run stopped.

For long ranges, increase the step size (e.g., `2–5s`) to reduce runtime and output length.

## 10. Quick Checklist

1. Run the agent or use previously cached results.
2. Review events in the Behavior Log → confirm/reject/edit.
3. Use embedding search to find & label similar frames.
4. Rerun with anchors if you’ve corrected key frames.
5. Switch to Classic Mode when you only want manual labeling.

## 11. Annolid Bot for Local Repo Tasks

In **AI Chat Studio**, Annolid Bot can inspect and modify your local Annolid workspace with file tools and repo-introspection helpers:

- `code_search(...)` to find where behavior, flags, or APIs are implemented.
- `code_explain(...)` to summarize Python modules/classes/functions and calls.
- `read_file(...)`, `write_file(...)`, `edit_file(...)` for targeted updates.

For safer edits, ask the bot to:

1. run `code_search` first,
2. explain the target with `code_explain`,
3. apply the smallest edit and report changed files/tests.
