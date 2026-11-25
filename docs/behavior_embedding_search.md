# Frame & Behavior Embedding Search (ONNX EmbeddingGemma 2)

Annolid supports fast, local, multimodal **frame and behavior search** powered by **ONNX EmbeddingGemma 2** (`onnx-community/embeddinggemma-2-ONNX`, Apache 2.0, ungated).

Because EmbeddingGemma 2 maps text, single video frames, and multi-frame video clips into a shared 768-dimensional vector space (with native Matryoshka truncation to 512, 256, or 128 dimensions), you can search videos by:

- **Natural-language behavior text** (e.g., `mouse grooming`, `nose-to-nose sniffing`, `rearing on hind legs`),
- **Reference frame similarity** (find frames visually/semantically matching the current frame),
- **Hybrid frame + text queries**,
- **Multi-frame temporal clips** (`4f` / `8f` windows) for dynamic behaviors, and
- **Zero-shot multi-behavior scoring** across a video timeline.

It runs via **ONNX Runtime** (`onnxruntime` + `tokenizers` + `huggingface-hub`) with **no PyTorch dependency** and downloads only the text backbone and vision encoder (~270 MB in `q4`, ~485 MB in `quantized` INT8), skipping the audio encoder.

---

## GUI Workflow (`Embedding Search` Dock)

1. Open a video in Annolid.
2. Enable **View / Tools → Frame Search (Embedding)** (`embeddingSearchDock`).
3. Keep the default backend **EmbeddingGemma 2 (ONNX)** (or switch to **DINOv3 (visual)** for image-only visual matching or **Qwen3-VL Embedding**).
4. Choose an ONNX preset and temporal window:
   - `Q4 · 140t · 768d` (default balanced speed & retrieval quality),
   - `Q4 · 70t · 256d (fast)` (~2–5× faster per frame on CPU with compact 256-dim vectors),
   - `Q4 · 280t · 768d (high)` (higher visual patch detail for fine postures),
   - `INT8 · 140t · 768d` or `FP32 · 140t · 768d`.
   - Temporal window: `1f (frame)`, `4f (clip)`, or `8f (clip)`.
5. Keep **Auto** enabled for adaptive similarity thresholding (calibrated for EmbeddingGemma 2's cosine range), or uncheck **Auto** to set a fixed `≥` threshold (typical EmbeddingGemma 2 values: `~0.66` for text-to-frame queries, `~0.82` for frame-to-frame queries; DINOv3 / Qwen3-VL: `~0.35`).
6. Run a search:
   - **Behavior text search:** type a behavior description in the query box and press **Enter** or click **Text**.
   - **Similar frame search:** navigate to an exemplar frame and click **Search**.
   - **Instant follow-up queries:** sampled frame embeddings are cached in memory per video/settings, so searching for additional behaviors or adjusting threshold/top-k re-queries in milliseconds.
   - **Zero-match diagnostics:** if no frame crosses the threshold, the status bar reports the best candidate frame and observed similarity range so you can tune the threshold immediately.
7. Double-click any result to jump directly to that frame, or select multiple results and click **Label Selected** to write behavior intervals into the ethogram/log.

---

## Similarity Thresholding & Soft-Token Speed Tuning

EmbeddingGemma 2 produces L2-normalized vectors in a higher, tighter cosine similarity cone than older CLIP models:

- **Text-to-frame:** unrelated frames typically score `~0.54–0.62`; matching frames typically score `~0.68–0.78`.
- **Frame-to-frame:** visually similar frames in the same recording typically score `~0.82–0.99`.

By default, omitting `--threshold` on the CLI (or leaving **Auto** checked in the GUI) computes a robust per-video adaptive threshold (`median + max(1.0 * 1.4826 * MAD, min_margin)` with backend-calibrated floors) and reports full `score_summary` statistics (`min`, `median`, `mean`, `p90`, `max`, `best_frame_index`, `effective_threshold`).

For faster CPU indexing and instant repeat queries across sessions:

- Lower the visual soft-token budget with `--soft-tokens 70` (default is `140`; `280` gives higher spatial resolution at ~2.4× the per-frame CPU cost of `140`).
- Increase `--stride` (e.g. `10`–`30`) on long videos and reuse embeddings via `--auto-cache` (bounded LRU `.npz` cache under `~/.cache/annolid/embedding_search`, configurable via `ANNOLID_EMBEDDING_CACHE_DIR`) or an explicit `--index-cache /path/to/index.npz`. The GUI `Embedding Search` dock automatically enables this bounded disk cache and reports live ETA progress during first-time indexing.
- Optionally tune ONNX Runtime CPU thread count via `ANNOLID_ONNX_THREADS=<N>`.

---

## Optional Audio Modality (Lazy On-Demand Download)

By default, text/image/video workflows download only `model*.onnx` and `vision_encoder*.onnx` (~270 MB in `q4`). If you call `EmbeddingGemma2OnnxAdapter.embed_audio(...)` / `embed_audios(...)` (or `predict(ModelRequest(task="embed", params={"audio": ...}))`), Annolid lazily downloads `audio_encoder*.onnx` (~189 MB in `q4`) on first audio use and embeds 16 kHz mono waveforms or `.wav` files into the exact same 768-d / Matryoshka semantic space:

```python
from annolid.core.models.adapters.embeddinggemma2_onnx import EmbeddingGemma2OnnxAdapter

with EmbeddingGemma2OnnxAdapter(quantization="q4") as adapter:
    audio_vec = adapter.embed_audio("/path/to/vocalization.wav")
    text_vec = adapter.embed_texts(["high pitched mouse squeak"])[0]
    print("cosine similarity:", float(audio_vec @ text_vec))
```

---

## CLI Workflow (`annolid-run behavior-search`)

Search a video by natural-language behavior description with automatic adaptive thresholding and cache the frame embeddings in a reusable `.npz` file:

```bash
annolid-run behavior-search /path/to/video.mp4 \
  --query "mouse grooming" \
  --quantization q4 \
  --soft-tokens 140 \
  --matryoshka-dim 256 \
  --stride 5 \
  --index-cache /path/to/video_eg2_index.npz \
  --out /path/to/grooming_matches.json \
  --json
```

Pass `--threshold` explicitly to override adaptive thresholding (e.g. `--threshold 0.68`), or `--soft-tokens 70` for fastest CPU indexing.

Reuse the cached `.npz` index for another behavior query in milliseconds:

```bash
annolid-run behavior-search /path/to/video.mp4 \
  --query "rearing on hind legs" \
  --quantization q4 \
  --soft-tokens 140 \
  --matryoshka-dim 256 \
  --stride 5 \
  --index-cache /path/to/video_eg2_index.npz \
  --json
```

Search using multi-frame temporal clips (`4` frames per window):

```bash
annolid-run behavior-search /path/to/video.mp4 \
  --query "chasing and fighting" \
  --clip-frames 4 \
  --stride 10 \
  --json
```

Zero-shot score sampled frames against multiple candidate behaviors:

```bash
annolid-run behavior-search /path/to/video.mp4 \
  --behaviors "grooming,rearing,sniffing,resting" \
  --stride 15 \
  --out /path/to/behavior_scores.json \
  --json
```

---

## Offline / Local ONNX Model Directory

You can pass a local directory containing the exported ONNX files (`onnx/model_q4.onnx`, `onnx/model_q4.onnx_data`, `onnx/vision_encoder_q4.onnx`, `onnx/vision_encoder_q4.onnx_data`, `tokenizer.json`, `config.json`, `processor_config.json`) via `--model-id /path/to/embeddinggemma-2-ONNX` or via the Python API:

```python
from annolid.services.embedding_search import search_video_behavior

result = search_video_behavior(
    "/path/to/video.mp4",
    query_text="mouse rearing",
    backend="embeddinggemma2_onnx",
    backend_params={
        "model_id": "/path/to/embeddinggemma-2-ONNX",
        "quantization": "q4",
        "image_max_soft_tokens": 140,
        "matryoshka_dim": 256,
        "local_files_only": True,
    },
    stride=5,
    threshold=None,  # adaptive per-video threshold (or pass a float e.g. 0.68)
)
print(result.score_summary, result.bouts)
```
