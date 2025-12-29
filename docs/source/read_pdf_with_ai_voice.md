# Read PDFs Aloud with AI Voice (Text-to-Speech)

Annolid includes an integrated PDF viewer with text-to-speech (TTS), so you can open a PDF and have selected text (or whole paragraphs) read aloud.

## Prerequisites

1) Install Annolid (see `install.md`).

2) Install PDF support:
```bash
pip install pymupdf
```

3) Install a TTS backend (pick one):

- Recommended (offline, higher-quality “AI voice”): Kokoro (ONNX)
```bash
pip install kokoro-onnx onnxruntime gdown
```

- Voice cloning (offline, uses a short voice prompt): Chatterbox Turbo (ONNX)
```bash
pip install onnxruntime soundfile
```
Then select `Engine = Chatterbox` and choose a voice prompt audio file in the `PDF Speech` dock (or edit `~/.annolid/tts_settings.json`).

- Language packs for Kokoro when you want Chinese or Japanese voices:
```bash
pip install misaki[zh]  # enables Mandarin (e.g., voice zf_001)
pip install misaki[ja]  # enables Japanese (e.g., voice jf_alpha)
```

- Fallback (online, simpler): Google TTS
```bash
pip install gTTS pydub
```
`pydub` needs `ffmpeg` available on your system.

## Open a PDF in Annolid

1) Launch the GUI:
```bash
annolid
```

2) Go to `File` → `Open PDF...` and pick a `.pdf`.

Annolid switches into PDF view and shows these docks (typically on the right):
- `PDF Speech` (voice / language / speed)
- `PDF Controls` (page + zoom)
- `PDF Reader` (click-to-read mode)

## Option A: Speak a selection (fastest)

This works in both the fallback viewer (image + text panel) and the PDF.js viewer.

1) Select some text (either in the page text panel, or directly on the PDF page).
2) Right-click → `Speak selection`.

## Option B: Click-to-read paragraphs (PDF.js reader mode)

This reads full paragraphs/sentences starting from where you click.

1) In the `PDF Reader` dock, enable `Use PDF.js (required for reader)`.
2) Keep `Enable click-to-read` turned on.
3) Click a paragraph in the PDF page to start reading.
4) Use `Pause/Resume`, `Stop`, `Prev`, `Next` in the same dock.

If the reader says it’s unavailable, install QtWebEngine (`pyqtwebengine` in conda, or `PyQtWebEngine` via pip) and restart Annolid.

## Change voice, language, and speed

Use the `PDF Speech` dock to set:
- `Voice` (example: `af_sarah`)
- `Voice` (Chinese): `zf_001` (requires `misaki[zh]`)
- `Voice` (Japanese): `jf_alpha` (requires `misaki[ja]`)
- `Language` (example: `en-us`)
- `Speed` (0.5–2.0)

These settings persist in `~/.annolid/tts_settings.json`.

## Troubleshooting

- **“PyMuPDF Required” dialog**: run `pip install pymupdf`.
- **No audio output**:
  - Make sure `ANNOLID_DISABLE_AUDIO` is not set.
  - On Linux servers/containers, ensure an audio device is present (or use a desktop machine).
- **First Kokoro run is slow**: Annolid downloads model files into `~/.annolid/kokoro` the first time.
- **gTTS fails**: it requires internet access; also ensure `ffmpeg` is installed for `pydub`.
