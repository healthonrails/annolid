# Image Editing (Local Diffusion / GGUF)

Annolid can generate or edit images using:

- **Diffusers (Python)**: e.g. `Qwen/Qwen-Image-2512`
- **stable-diffusion.cpp (`sd-cli`)**: e.g. GGUF models like `unsloth/Qwen-Image-2512-GGUF`

## Install (Diffusers)

```bash
pip install -U "annolid[image_editing]"
```

## CLI examples

### Diffusers (text-to-image)

```bash
annolid-run predict image-edit \
  --backend diffusers \
  --model-id Qwen/Qwen-Image-2512 \
  --prompt "cartoon sloth wearing a lab coat" \
  --width 1024 --height 1024 \
  --output qwen.png
```

### stable-diffusion.cpp (GGUF preset)

1) Build or download `stable-diffusion.cpp` and locate the `sd-cli` binary.

2) Run with the built-in preset (downloads weights from Hugging Face on first run):

```bash
annolid-run predict image-edit \
  --backend sdcpp \
  --sd-cli /path/to/sd-cli \
  --preset qwen-image-2512-gguf \
  --quant Q2_K \
  --prompt "cartoon sloth" \
  --width 1024 --height 1024 \
  --output qwen_gguf.png
```

Notes:

- `--preset`/`--quant` are **Annolid** options (they are not valid flags for `sd-cli` itself).
- `Q2_K` is the repo naming; Annolid also accepts `Q2-K` and normalizes it to `Q2_K`.
- On macOS Metal builds, if you see `unsupported op 'DIAG_MASK_INF'`, update/rebuild `stable-diffusion.cpp` (or build a CPU-only `sd-cli` with Metal disabled, e.g. `cmake -B build-cpu -DGGML_METAL=OFF`).

## GUI

Open **File → Image Editing…** (or the toolbar button), choose a backend, enter a prompt, and click **Run**.
