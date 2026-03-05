# Installation

Annolid supports Python `3.10` to `3.13`.

## Recommended (uv)

`uv` provides fast, reproducible environment setup.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install Annolid:

```bash
uv pip install annolid
```

Install from source (development):

```bash
git clone https://github.com/healthonrails/annolid.git
cd annolid
uv pip install -e .
```

## pip Alternative

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install annolid
```

## Verify Installation

```bash
annolid --help
annolid-run --help
```

## Related Guides

- Existing detailed install docs: [docs/installation.md](https://github.com/healthonrails/annolid/blob/main/docs/installation.md)
- One-line install choices: [docs/one_line_install_choices.md](https://github.com/healthonrails/annolid/blob/main/docs/one_line_install_choices.md)
- uv-focused setup: [docs/install_with_uv.md](https://github.com/healthonrails/annolid/blob/main/docs/install_with_uv.md)
