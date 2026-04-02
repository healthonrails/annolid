# Getting Started

Use this page as the fastest path from install to a working Annolid session.

## 20-Minute Path

1. Install Annolid with your preferred method from [Installation](installation.md).
2. Confirm commands resolve:
   - `annolid --help`
   - `annolid-run --help`
3. Launch the GUI with `annolid`.
4. Open a short sample video.
5. Label one or more instances on a representative frame.
6. Run tracking/propagation and review identity consistency.
7. Export results for downstream analysis.

If this flow works, your environment is ready for full projects.

## Quick Environment Check

Recommended development setup in this repository:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[gui]"
```

Validate:

```bash
annolid --help
annolid-run list-models
```

## First GUI Session Checklist

- Open video
- Pick representative frame
- Add labels/shapes/keypoints
- Track forward/backward
- Fix errors early (occlusion, swaps, overlap)
- Save and export

See [Workflows](workflows.md) for detailed operating patterns.

## First CLI Session Checklist

```bash
annolid-run --help
annolid-run list-models
annolid-run help train
annolid-run help predict
annolid-run help train <model>
annolid-run help predict <model>
```

Use run-config templates under `annolid/configs/runs/` for reproducible jobs.

For built-in models, the model help view now shows a grouped quick reference
before the full option list.

If you want Annolid Bot to drive those commands for you, use the typed
`annolid_run` path described in [Annolid Agent and annolid-run](agent_annolid_run.md).

## Optional Next Layers

- Memory subsystem for reusable context: [Memory Subsystem](memory.md)
- Agent integrations and MCP tools: [MCP](mcp.md)
- Security hardening for agents/secrets: [Agent Security](agent_security.md)
- Behavior workflow tutorial (Timeline + Flags + Annolid Bot): [Behavior labeling with Timeline, Flags, and Annolid Bot](tutorials/behavior_timeline_flags_bot.md)
- Practical tutorials and notebooks: [Tutorials](tutorials.md)

## Troubleshooting

- `qtpy.QtBindingsNotFoundError`: install with `.[gui]` extra.
- Missing codecs or video issues: install FFmpeg for your platform.
- Plugin/model command errors: check active environment and run `annolid-run --help`.
