# Agent and Automation

Use this section when you are configuring Annolid Bot, connecting external tools,
or running repeatable workflows through the agent stack.

## Start Here

<div class="ann-grid">
  <article class="ann-card">
    <h3>Agent CLI</h3>
    <p>Run Annolid-native CLI flows through the typed <code>annolid_run</code> path.</p>
    <a href="agent_annolid_run/">Open agent CLI guide</a>
  </article>
  <article class="ann-card">
    <h3>MCP</h3>
    <p>Connect external tools and resources through Model Context Protocol servers.</p>
    <a href="mcp/">Open MCP guide</a>
  </article>
  <article class="ann-card">
    <h3>Codex and ACP</h3>
    <p>Bridge Annolid with Codex-style workflows and ACP-compatible runtime paths.</p>
    <a href="codex_and_acp/">Open Codex and ACP guide</a>
  </article>
  <article class="ann-card">
    <h3>Calendar</h3>
    <p>Schedule and coordinate tasks with Google Calendar-aware agent flows.</p>
    <a href="agent_calendar/">Open calendar guide</a>
  </article>
  <article class="ann-card">
    <h3>Workspace and Secrets</h3>
    <p>Configure Google Workspace, local secret storage, and channel-safe credentials.</p>
    <a href="agent_workspace/">Open workspace guide</a>
  </article>
  <article class="ann-card">
    <h3>Memory and Security</h3>
    <p>Manage retrieval-backed memory and harden agent behavior before scaling up.</p>
    <a href="agent_security/">Open security guide</a>
  </article>
</div>

## Recommended Sequence

1. Start with [Annolid Agent and annolid-run](agent_annolid_run.md) for safe CLI execution.
2. Configure [MCP](mcp.md) if you need external tools or browser/file bridges.
3. Set up [Google Workspace](agent_workspace.md) and [Agent Secrets](agent_secrets.md) before enabling integrations.
4. Review [Agent Security](agent_security.md) and [Memory Subsystem](memory.md) before turning on broader automation.

## What Lives Here

- typed agent tool execution,
- MCP connectivity,
- Codex and ACP integration notes,
- calendar and workspace integrations,
- secret handling,
- memory-backed agent behavior,
- security and operational guardrails.
