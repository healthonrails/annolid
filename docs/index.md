<section class="ann-hero">
  <span class="ann-kicker">Annolid Documentation Portal</span>
  <h1>Ship annotation and tracking workflows faster.</h1>
  <p>
    Annolid combines a desktop GUI, a composable CLI, and agent-assisted tools
    for real-world behavior analysis projects.
  </p>
  <p>
    Start with a stable local setup, then move into repeatable workflows,
    memory-backed context, and deployment-ready operations.
  </p>
  <div class="ann-quick">
    <a href="getting_started.md">Getting Started in 20 minutes</a>
    <a href="installation.md">Install Annolid</a>
    <a href="install_with_uv.md">Use <code>.venv</code> for development</a>
    <a href="workflows.md">Follow core workflows</a>
    <a href="agent_annolid_run.md">Run <code>annolid-run</code> from Annolid agents</a>
    <a href="tutorials.md">Explore tutorials</a>
    <a href="memory.md">Enable memory subsystem</a>
  </div>
</section>

## Choose Your Path

<div class="ann-paths">
  <article class="ann-path">
    <strong>New Users</strong>
    Start from installation and one-line setup paths, then run your first GUI project.
    <br />
    <a href="installation.md">Installation</a> |
    <a href="one_line_install_choices.md">One-Line Installer</a>
  </article>
  <article class="ann-path">
    <strong>Power Users</strong>
    Use <code>annolid-run</code>, MCP integrations, and automation-friendly CLI flows.
    <br />
    <a href="workflows.md">Workflows</a> |
    <a href="agent_annolid_run.md">Agent CLI</a> |
    <a href="mcp.md">MCP</a> |
    <a href="reference.md">Reference</a>
  </article>
  <article class="ann-path">
    <strong>Maintainers</strong>
    Keep deployment, migration, and release workflows predictable and documented.
    <br />
    <a href="deployment.md">Deployment</a> |
    <a href="migration.md">Migration Plan</a>
  </article>
</div>

## Core Areas

<div class="ann-grid">
  <article class="ann-card">
    <h3>Setup</h3>
    <p>Pick the right environment path for your platform and development mode.</p>
    <a href="installation.md">Open installation guide</a>
  </article>
  <article class="ann-card">
    <h3>Workflow Execution</h3>
    <p>Run GUI and CLI tasks with explicit, reproducible command patterns.</p>
    <a href="workflows.md">Open workflows</a>
  </article>
  <article class="ann-card">
    <h3>Agent CLI</h3>
    <p>Use the typed <code>annolid_run</code> tool for safe agent-driven CLI operations.</p>
    <a href="agent_annolid_run.md">Open agent CLI guide</a>
  </article>
  <article class="ann-card">
    <h3>Tutorials</h3>
    <p>Jump into focused guides for tracking, segmentation, and model operations.</p>
    <a href="tutorials.md">Open tutorials</a>
  </article>
  <article class="ann-card">
    <h3>Memory System</h3>
    <p>Store reusable context, use scoped retrieval, and migrate legacy memory data.</p>
    <a href="memory.md">Open memory docs</a>
  </article>
  <article class="ann-card">
    <h3>Agents and Security</h3>
    <p>Configure agents, isolate secrets, and validate local security posture.</p>
    <a href="agent_security.md">Open security docs</a>
  </article>
  <article class="ann-card">
    <h3>Operations</h3>
    <p>Deploy docs/site assets and keep release and migration flows in sync.</p>
    <a href="deployment.md">Open deployment guide</a>
  </article>
</div>

## Product Snapshot

- Python package metadata supports `>=3.10`; CI/docs currently target Python 3.10-3.13.
- Primary entry points are `annolid` (GUI) and `annolid-run` (CLI/plugins).
- Memory subsystem includes GUI CRUD manager, structured settings profiles, and legacy-source migration tooling.
- Annolid Bot supports multimodal chat and optional provider integrations in the GUI.
- Docs are built with MkDocs Material in strict mode and published through GitHub Actions.
