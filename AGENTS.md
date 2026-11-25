# AGENTS.md — Annolid Engineering & Coding-Agent Standard

Required standard for coding agents and contributors working in the Annolid repository.
Annolid supports real research and lab workflows: every change must protect **correctness, reproducibility, user data integrity, and workflow continuity** across GUI, CLI, annotation, tracking, behavior, and agent pipelines.

Priority order (never trade higher items for lower ones):
**1. Correctness → 2. Reproducibility → 3. Data Integrity → 4. Reliability → 5. Usability → 6. Maintainability → 7. Performance**

---

## 1. Hard Rules (Non-Negotiable)

1. **Fix at the true boundary, not the symptom.** Choose the *minimum necessary change set* that removes the root cause cleanly — neither a fragile local patch nor unrelated cleanup.
2. **Protect external contracts by default.** CLI flags, config keys/defaults, LabelMe/COCO/BORIS/CSV schemas, public Python APIs, saved project behavior, and GUI workflows are compatibility-sensitive. Prefer: *additive support → compatibility shim → deprecation warning → removal (only when justified and documented)*.
3. **Never touch vendored third-party trees** for cleanup, formatting, or speculative edits (see §3).
4. **Never commit secrets, tokens, credentials, model weights, or machine-local absolute paths** (e.g., `/Users/<name>/...`).
5. **Never claim checks passed unless actually run after the final edit.** Always state exactly what was run, what was skipped, and why.
6. **Use `.venv` for local validation** unless the task explicitly requires another environment.
7. **Do not swallow critical failures.** No bare `except:`, no silent fallbacks that mask data loss or corruption, no hidden retries.
8. **Keep changes and commits scoped.** One logical change per commit; do not mix unrelated refactors, repo-wide formatting churn, and feature work.

---

## 2. Repository Map & Layering

**Entrypoints**
- `annolid` → `annolid/gui/launcher.py` → `annolid/gui/app.py` (`annolid/gui/window_base.py`)
- `annolid-run` → `annolid/engine/cli.py` (delegates `colab`, `sim2real`, `runs`, `runtime`, `memory`, `train`, `predict`)
- Legacy CLI helper → `annolid/main.py`

**4-Layer Architecture** (canonical reference: `docs/architecture_layers.md`)
- `annolid/domain/` — framework-agnostic schemas, data types, invariants.
- `annolid/services/` — orchestration use-cases shared by GUI, CLI, and workers (e.g., `tracking`, `training_monitor`, `training_monitor_cli`, `behavior_analytics`, `zone_analytics`, `time_budget`, `embedding_search`, `chat_*`).
- `annolid/interfaces/` — user/transport adapters (GUI, CLI, bots, background jobs).
- `annolid/infrastructure/` — filesystem, persistence, capabilities (`capabilities.py`), sandbox execution, LanceDB memory, external API adapters.
- **Layer rules:** Interfaces call `services`/`infrastructure` facades, not each other and not `annolid.core.agent` directly from GUI widgets/dialogs; `domain` stays framework-agnostic; `infrastructure` never imports GUI code.

**Key Subsystems**
- `annolid/gui/` — `window_base.py`, `mixins/` (lifecycle, persistence, canvas, playback, label panel, shape editing), `widgets/` (`canvas.py`, `flags.py`, `timeline_panel.py`, dialogs), `controllers/`, `features/`.
- `annolid/annotation/` & `annolid/io/` — LabelMe/COCO/YOLO/CSV/pose converters, `shape_io.py`, `svg_import.py`, `large_image/`.
- `annolid/utils/` — `annotation_store.py` (`*_annotations.ndjson`), `atomic_io.py`, `safe_torch_load.py`, `llm_settings.py`, `devices.py`.
- `annolid/tracker/`, `annolid/tracking/`, `annolid/segmentation/`, `annolid/postprocessing/` — point/VOS/SAM/YOLO trackers, pose filtering/evaluation, tracking reports.
- `annolid/behavior/`, `annolid/colab/`, `annolid/sim2real/`, `annolid/core/agent/` — BORIS/IRR/TCN/cohort analytics, remote Colab execution, Sim-to-Real studio, Annolid Bot & tools.

---

## 3. Vendored / Do-Not-Refactor Trees

Unless a task explicitly targets a bug in an Annolid wrapper inside one of these directories, **do not refactor, reformat, or run broad edits inside**:
- `annolid/detector/countgd/`
- `annolid/detector/yolov5/`
- `annolid/segmentation/SAM/` (upstream SAM/SAM2/SAM3 internals)
- `annolid/segmentation/cutie_vos/` (upstream Cutie internals)
- `annolid/segmentation/yolact/`
- `annolid/segmentation/MEDIAR/`
- `annolid/depth/video_depth_anything/`
- `annolid/tracker/cowtracker/cowtracker/thirdparty/`
- `annolid/gui/assets/pdfjs/`

---

## 4. Annolid-Specific Invariants & Footguns

### 4.1 Lean `annolid[gui]` & Frozen Bundle Boundaries
- Core entrypoints (`annolid.main`, `annolid.engine.cli`, `annolid.gui.launcher`, `annolid.gui.app`, `annolid.gui.window_base`, `annolid.core.agent.tools`, `annolid.colab.service`, `annolid.sim2real`, `annolid.services.training_monitor_cli`) **must import cleanly without optional extras** (`torch`, `torchvision`, `ultralytics`, `matplotlib`, `scipy`, `sklearn`, `skimage`, `h5py`, `onnxruntime`, `transformers`, `diffusers`, `lancedb`, `fitz`, `gdown`, `tensorboard`, `openai`, `anthropic`, `sounddevice`, `librosa`, `pyvips`, `openslide`, `ffpyplayer`).
- Keep `annolid/core/agent/tools/__init__.py` PEP 562 lazy (`__getattr__`).
- Frozen desktop bundles (`annolid.spec` + `scripts/check_distribution_artifacts.py`) intentionally exclude ML stacks and `pandas`. Do not add eager top-level `pandas` or `matplotlib` imports to `annolid/gui/` modules; import lazily inside methods.
- Whenever modifying dependencies/extras in `pyproject.toml`, keep `annolid/infrastructure/capabilities.py`, `annolid.spec` (if runtime assets change), and `uv.lock` synchronized.

### 4.2 Data Integrity & Dual Annotation Storage
- Video projects can store frames in **both** per-frame disk `.json` files (manual seeds) and `AnnotationStore` (`*_annotations.ndjson`, when `persist_json=False`). Any code reading, filtering, reporting, or deleting video annotations must handle both sources consistently.
- Preserve LabelMe round-trip fidelity (`imagePath`, optional `imageData`, `point_labels`, `flags`, `otherData`, non-destructive display previews). Never delete standalone user source images when deleting a label file.
- Use atomic writes (`atomic_write_json`, `atomic_write_text`, or `tempfile` + `os.replace`) for state files, caches, credentials, and JSON/NDJSON exports.

### 4.3 GUI & Qt (`qtpy` / PySide6) Rules
- Annolid defaults to `PySide6-Essentials` via `qtpy`.
- Validate: dirty state, selection/visibility/checkbox state, file/frame switches, unsaved-change prompts, load/save transitions, Undo/Redo sync, and UI responsiveness.
- **PySide6 checkbox gotcha:** `QCheckBox.stateChanged` emits an `int`, which does not reliably equal `Qt.CheckState.Checked` in PySide6 — use `checkbox.isChecked()` instead of `state == Qt.Checked`.
- **Shortcuts:** Scope dock/panel `QShortcut`s with `Qt.WidgetWithChildrenShortcut` so they never hijack main-window `Ctrl+Z` / `Ctrl+S` / `Delete` shortcuts.
- **Threading:** Never block the UI thread with network calls, package installs, or heavy loops; never mutate Qt widgets from worker threads; ensure worker `finally:` cleanup and `finished` signals fire even on cancel/error.

### 4.4 ML, Tracking & Scientific Correctness
- **Frame ranges:** `CV2Video.get_frames_between(start, end)` is half-open `[start, end)`, whereas tracker/UI `end_frame` is often inclusive or `-1` (meaning `total_frames - 1`). Normalize `end_frame` explicitly at entry boundaries.
- **PyTorch hygiene:** Never call global `torch.set_grad_enabled(False)` (breaks training in the same session); scope inference with `with torch.inference_mode():`. Always load checkpoints via `annolid.utils.safe_torch_load.safe_torch_load(path, map_location=...)` (never raw `torch.load`, and do not pass `weights_only=`). Honor explicit `device="cpu"` requests.
- **No data leakage / silent distortion:** Split train/val by `source_group` before oversampling; mask padded sequence steps (`-100`) in TCN training; split temporal smoothing across non-contiguous frame gaps; use word-boundary regexes (not bare substrings) for behavior/event classification.

### 4.5 Security & Agent Sandboxing
- Enforce `allowed_dir` / `allowed_read_roots` on `Path(...).expanduser().resolve()` for **all** agent-accessible inputs — including paths referenced inside YAML configs, `.txt` manifests, symlinks, inline `--flag=/path` CLI arguments, and `~`-prefixed weight paths.
- Sanitize archive members, Drive output names, and COCO `file_name` paths against `..` traversal and absolute paths before writing.
- Never run untrusted or LLM-generated code via in-process `exec()`; keep generated analysis inside the AST-checked subprocess sandbox (`annolid/infrastructure/sandbox/execution.py`).
- Enforce URL scheme checks, response/frame byte caps, subprocess/socket timeouts, and restrictive file permissions (`0o600` via atomic temp file) on credentials and tokens.

---

## 5. Efficient Agent Execution Workflow

For every non-trivial task, work in tight, batched phases:

1. **Frame before editing:** State (internally or briefly) the user-visible goal, true root cause & code path, invariants to preserve, blast radius, and testable acceptance target. Ask the user for clarification **only** when genuinely blocking; otherwise proceed with stated assumptions.
2. **Explore in parallel:** Batch independent searches and full/targeted file reads in the same turn. Avoid re-reading unchanged files or probing vendored trees unnecessarily.
3. **Edit cleanly at the right boundary:** Batch independent edits in parallel. Keep internal implementations clean while preserving external contracts. Update user-facing docs (`README.md`, `docs/`, `RELEASING.md`) in the same change set whenever behavior or CLI/install workflows change.
4. **Add focused regression coverage:** Write deterministic unit tests targeting the exact failure mode. Isolate `HOME`, settings, and cache directories with `tmp_path` + `monkeypatch` so tests never touch `~/.annolid` or `~/.config/colab-cli`, and avoid live network or unguarded loopback dependencies.
5. **Validate proportionally (smallest meaningful check first, then expand):** See §6. Fix all reported errors before re-running; do not re-run already-passing suites when no relevant code changed.

---

## 6. Validation Matrix

Always activate `.venv` (or invoke `.venv/bin/...` directly) and run GUI tests headless:

```bash
export QT_QPA_PLATFORM=offscreen
export ANNOLID_DISABLE_BOT_AUTOSTART=1
```

| Surface Touched | Required Validation Commands |
| --- | --- |
| **Any Python change** | `.venv/bin/ruff format --check --config pyproject.toml annolid tests scripts`<br>`.venv/bin/ruff check --config pyproject.toml annolid tests scripts`<br>`.venv/bin/pytest -q <targeted_test_files>` |
| **Core / GUI / CLI / agent tools imports** | `.venv/bin/pytest -q tests/test_repo_review_fixes.py -k lean_install` (verifies zero optional-extra leakage) |
| **GUI widgets / dialogs / mixins** | Targeted GUI tests + check no eager `pandas`/`matplotlib` top-level imports |
| **Dependencies / `pyproject.toml` / `annolid.spec` / release** | `uv lock --check`<br>`.venv/bin/python -m build --no-isolation --outdir /tmp/annolid_dist_check`<br>`.venv/bin/python scripts/check_distribution_artifacts.py /tmp/annolid_dist_check/*.whl /tmp/annolid_dist_check/*.tar.gz`<br>`.venv/bin/python -m twine check /tmp/annolid_dist_check/*.whl /tmp/annolid_dist_check/*.tar.gz`<br>*(clean `/tmp/annolid_dist_check` afterward)* |
| **Shell scripts / installers** | `bash -n install.sh scripts/release.sh scripts/sync_docs_to_site_root.sh` |
| **Docs / `mkdocs.yml` / `website/`** | `.venv/bin/python scripts/site_workflow.py check`<br>`NO_MKDOCS_2_WARNING=1 .venv/bin/mkdocs build --strict --clean --config-file mkdocs.yml -d /tmp/annolid_mkdocs_check && rm -rf /tmp/annolid_mkdocs_check` |
| **High-risk / multi-subsystem / pre-release** | `source .venv/bin/activate && pytest --ignore=tests/test_sam3_prompt_transaction_contract.py` + `pytest tests/test_sam3_prompt_transaction_contract.py` + `pre-commit run --all-files` |

Notes for pre-commit: hooks enforce `ruff check`, `ruff format`, trailing whitespace, EOF newline, YAML/TOML/JSON validity, large-file limits (`<=6 MB`), `codespell` (avoid typo-like short identifiers), and `gitleaks`.

---

## 7. Commit & Handoff Contract

### Commits
Use scoped Conventional Commits — one logical change per commit, intent-clear subject, and concise body explaining *why* for non-trivial changes:
- `fix(gui): restore shape visibility defaults on reload`
- `fix(io): preserve optional label fields during round-trip serialization`
- `fix(security): enforce allowed_read_roots in Colab dataset packaging`
- `feat(cli): add headless training monitor runs subcommand`
- `docs(install): clarify lean GUI profile and ONNX GPU setup`
- `ci(docs): gate site sync jobs on changed surfaces`

Never commit unless asked, and never push to a remote without explicit user instruction.

### Final Handoff
Be concise, direct, and technically precise. Separate verified facts from inference, and include:
1. **Changed files** and **behavioral impact** (what changed for users/developers and why).
2. **Exact validation performed** (commands and actual pass/fail/skip counts).
3. **Skipped checks, assumptions, and residual risks** (never conceal uncertainty or environment blocks).
