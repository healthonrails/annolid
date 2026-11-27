from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from annolid.behavior.labeling_skill import behavior_labeling_prompt_guidance
from annolid.core.agent.skill_registry.schema import validate_skill_manifest
from annolid.core.agent.skills import AgentSkillsLoader
from annolid.services.behavior_agent import list_behavior_subagent_profiles


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILTIN_SKILLS_DIR = REPO_ROOT / "annolid" / "core" / "agent" / "skills"

_TOOL_PREFIXES = (
    "gui_",
    "annolid_",
    "video_",
    "mcp_browser",
    "bibtex_",
    "clawhub_",
    "coding_session_",
    "admin_",
    "sam3_",
    "sim2real_",
    "google_",
    "git_",
    "gh_",
    "github_",
    "exec_",
    "memory_",
    "web_",
    "extract_pdf_",
    "download_",
    "search_literature",
    "build_research_packet",
    "draft_paper_swarm",
    "assess_paper_draft",
    "automation_schedule",
    "spawn_behavior_subagent",
    "run_swarm",
)
# Parameter names in tool signatures that happen to start with a tool prefix.
_NON_TOOL_PARAM_NAMES = frozenset({"video_folder", "video_path", "video_description"})


def _collect_registered_tool_names() -> set[str]:
    names: set[str] = set()
    roots = (
        REPO_ROOT / "annolid" / "core" / "agent" / "tools",
        REPO_ROOT / "annolid" / "core" / "agent" / "gui_backend",
        REPO_ROOT / "annolid" / "gui" / "widgets",
    )
    for root in roots:
        if not root.exists():
            continue
        for py_path in sorted(root.rglob("*.py")):
            source = py_path.read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "name":
                    for sub in ast.walk(node):
                        if (
                            isinstance(sub, ast.Return)
                            and isinstance(sub.value, ast.Constant)
                            and isinstance(sub.value.value, str)
                        ):
                            names.add(sub.value.value)
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if (
                            kw.arg == "name"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            names.add(kw.value.value)
    return names


def test_all_builtin_skills_have_valid_manifests_matching_names_and_tags(
    tmp_path: Path,
) -> None:
    loader = AgentSkillsLoader(tmp_path)
    rows = loader.list_skills(filter_unavailable=False)
    assert len(rows) >= 40

    for row in rows:
        name = row["name"]
        assert row["manifest_valid"] is True, f"{name}: {row['manifest_errors']}"
        assert str(row.get("description") or "").strip(), f"{name}: missing description"
        frontmatter = loader.get_skill_metadata(name) or {}
        assert frontmatter.get("name") == name, (
            f"{name}: frontmatter name {frontmatter.get('name')!r} != folder slug"
        )
        parsed_meta = loader._parse_skill_meta(frontmatter)
        assert parsed_meta.get("tags"), f"{name}: expected non-empty tags for retrieval"

        raw_metadata = frontmatter.get("metadata")
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            payload = json.loads(raw_metadata)
            assert not ("annolid" in payload and "requires" in payload), (
                f"{name}: 'requires' must be nested inside 'annolid' metadata object"
            )

    pool = loader.describe_skill_pool()
    assert pool["counts"]["total"] == len(rows)
    assert pool["preview_truncated"] == 0
    assert len(pool["preview"]) == len(rows)
    assert "web-scraping" not in pool["always_skills"]


def test_all_behavior_subagent_profile_skills_exist_and_load(tmp_path: Path) -> None:
    loader = AgentSkillsLoader(tmp_path)
    builtin_names = {
        row["name"] for row in loader.list_skills(filter_unavailable=False)
    }

    for profile in list_behavior_subagent_profiles():
        assert profile.default_skill_names, (
            f"{profile.name}: missing default_skill_names"
        )
        for skill_name in profile.default_skill_names:
            assert skill_name in builtin_names, (
                f"{profile.name}: references missing built-in skill {skill_name!r}"
            )
            loaded = loader.load_skills_for_context([skill_name])
            assert f"### Skill: {skill_name}" in loaded


def test_builtin_skills_only_reference_registered_tool_names() -> None:
    registered_tools = _collect_registered_tool_names()
    assert "gui_context" in registered_tools
    assert "google_colab" in registered_tools
    assert "sim2real_workflow" in registered_tools

    unknown_refs: list[tuple[str, str]] = []
    for skill_file in sorted(BUILTIN_SKILLS_DIR.glob("*/SKILL.md")):
        content = skill_file.read_text(encoding="utf-8")
        for match in re.finditer(r"`([a-z][a-z0-9_]+)(?:\(|`)", content):
            token = match.group(1)
            if token in _NON_TOOL_PARAM_NAMES:
                continue
            if (
                any(token.startswith(prefix) for prefix in _TOOL_PREFIXES)
                and token not in registered_tools
            ):
                unknown_refs.append((skill_file.parent.name, token))

    assert unknown_refs == [], f"Unknown tool references in SKILL.md: {unknown_refs}"


def test_behavior_labeling_prompt_guidance_reads_contract_from_skill() -> None:
    guidance = behavior_labeling_prompt_guidance()
    assert "strict JSON" in guidance
    assert "no_behavior" in guidance
    assert "Ground the description in visible frame-grid evidence" in guidance


def test_skill_manifest_validates_tags_and_keywords() -> None:
    ok_result = validate_skill_manifest(
        {
            "name": "demo",
            "description": "Demo skill",
            "metadata": '{"annolid":{"tags":["alpha","beta"],"keywords":"gamma"}}',
        }
    )
    assert ok_result.valid is True

    bad_nested = validate_skill_manifest(
        {
            "name": "demo",
            "description": "Demo skill",
            "metadata": '{"annolid":{"tags":[123]}}',
        }
    )
    assert bad_nested.valid is False
    assert any(
        "tags entries must be non-empty strings" in err for err in bad_nested.errors
    )

    bad_top_level = validate_skill_manifest(
        {
            "name": "demo",
            "description": "Demo skill",
            "keywords": 42,
        }
    )
    assert bad_top_level.valid is False
    assert any(
        "keywords must be a string or list of strings" in err
        for err in bad_top_level.errors
    )


def test_suggest_skills_excludes_disable_model_invocation_and_uses_tags(
    tmp_path: Path,
) -> None:
    hidden_dir = tmp_path / "skills" / "hidden-secret"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    (hidden_dir / "SKILL.md").write_text(
        "---\n"
        "name: hidden-secret\n"
        "description: Handles zebra telemetry diagnostics\n"
        "disable-model-invocation: true\n"
        "---\n"
        "Do not auto-suggest.\n",
        encoding="utf-8",
    )

    tagged_dir = tmp_path / "skills" / "telemetry-helper"
    tagged_dir.mkdir(parents=True, exist_ok=True)
    (tagged_dir / "SKILL.md").write_text(
        "---\n"
        "name: telemetry-helper\n"
        "description: Inspect experiment logs\n"
        'metadata: \'{"annolid":{"tags":["zebra telemetry","qc"]}}\'\n'
        "---\n"
        "Inspect telemetry.\n",
        encoding="utf-8",
    )

    loader = AgentSkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
    suggestions = loader.suggest_skills_for_task_scored(
        "Run zebra telemetry qc on the latest run", top_k=5
    )
    names = [row["name"] for row in suggestions]
    assert "hidden-secret" not in names
    assert names == ["telemetry-helper"]


def test_embedding_retrieval_indexes_skill_tags(tmp_path: Path) -> None:
    alpha_dir = tmp_path / "skills" / "alpha-tool"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    (alpha_dir / "SKILL.md").write_text(
        "---\n"
        "name: alpha-tool\n"
        "description: General utility helper\n"
        'metadata: \'{"annolid":{"tags":["mujoco","flybody"]}}\'\n'
        "---\n"
        "Body.\n",
        encoding="utf-8",
    )
    beta_dir = tmp_path / "skills" / "beta-tool"
    beta_dir.mkdir(parents=True, exist_ok=True)
    (beta_dir / "SKILL.md").write_text(
        "---\nname: beta-tool\ndescription: General utility helper\n---\nBody.\n",
        encoding="utf-8",
    )

    class _TagAwareEmbedder:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            del normalize_embeddings, show_progress_bar
            vectors = []
            for text in texts:
                lowered = str(text or "").lower()
                if "mujoco" in lowered or "flybody" in lowered:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    loader = AgentSkillsLoader(
        tmp_path,
        builtin_skills_dir=tmp_path / "builtin",
        skill_retrieval_mode="embedding",
        embedding_client=_TagAwareEmbedder(),
    )
    assert loader.suggest_skills_for_task("Run mujoco simulation", top_k=1) == [
        "alpha-tool"
    ]


def test_builtin_skill_retrieval_quality_for_core_and_new_workflows(
    tmp_path: Path,
) -> None:
    loader = AgentSkillsLoader(tmp_path)

    cases: list[tuple[str, str, bool]] = [
        ("Schedule an email reminder for tomorrow at 9am", "cron", True),
        (
            "Train a TCN polygon classifier on mouse behavior annotations",
            "polygon-classifier-training",
            True,
        ),
        (
            "Run track qc and inter-rater reliability on our ethogram",
            "behavior-evaluation",
            False,
        ),
        ("Train YOLO on Google Colab L4 GPU and pull results", "google-colab", True),
        ("Generate synthetic Mouse 3D sim2real training data", "sim2real", True),
        (
            "Relabel nose to snout across all frames in mouse_annotations.ndjson",
            "shape-annotation-store",
            False,
        ),
        (
            "Relabel nose to snout across all frames in mouse_annotations.ndjson",
            "shape-workflows",
            False,
        ),
        ("Upload saved videos to Cornell Box or Google Drive", "cloud-workspace", True),
        (
            "Check novelty of our research paper draft and verify citations",
            "research-writing",
            False,
        ),
        (
            "Fine-tune a YOLO detection model on our labeled dataset",
            "model-training",
            True,
        ),
        (
            "Delegate a multi-agent roundtable swarm to review architecture",
            "multi-agent-orchestration",
            True,
        ),
    ]

    for query, expected_skill, require_top1 in cases:
        suggestions = loader.suggest_skills_for_task_scored(query, top_k=5)
        names = [row["name"] for row in suggestions]
        if require_top1:
            assert names and names[0] == expected_skill, (
                f"Expected {expected_skill!r} as top-1 for {query!r}, got {names}"
            )
        else:
            assert expected_skill in names, (
                f"Expected {expected_skill!r} in top-5 for {query!r}, got {names}"
            )


def test_web_scraping_auto_retrieval_and_no_open_field_false_positives(
    tmp_path: Path,
) -> None:
    loader = AgentSkillsLoader(tmp_path)
    web_queries = [
        "Summarize this page",
        "What is on the current page?",
        "Read the open page and extract the table",
        "Open https://example.com and scrape the pricing",
        "Fill out the form on this website and click submit",
        "Take a screenshot of the open browser tab",
        "Extract structured fields from this webpage",
        "Save this page to the workspace",
        "Check what URL is open right now",
        "Fetch https://docs.python.org/3/ and summarize the page",
        "Inspect the DOM of the open page",
    ]
    for query in web_queries:
        names = [
            row["name"] for row in loader.suggest_skills_for_task_scored(query, top_k=3)
        ]
        assert "web-scraping" in names, (
            f"Expected 'web-scraping' in top-3 for {query!r}, got {names}"
        )

    for open_ui_query in (
        "Open https://example.com and scrape the pricing",
        "Take a screenshot of the open browser tab",
        "Check what URL is open right now",
        "Inspect the DOM of the open page",
    ):
        names = [
            row["name"]
            for row in loader.suggest_skills_for_task_scored(open_ui_query, top_k=3)
        ]
        assert "behavior-assay-taxonomy" not in names, (
            f"Unexpected 'behavior-assay-taxonomy' false positive for {open_ui_query!r}: {names}"
        )

    open_field_names = [
        row["name"]
        for row in loader.suggest_skills_for_task_scored(
            "Analyze open field center vs periphery occupancy", top_k=3
        )
    ]
    assert "behavior-assay-taxonomy" in open_field_names
    assert "metrics-derivation" in open_field_names


def test_default_system_prompt_fits_bootstrap_and_all_builtin_skills_without_truncation(
    tmp_path: Path,
) -> None:
    from annolid.core.agent.context import AgentContextBuilder
    from annolid.core.agent.workspace_bootstrap import bootstrap_workspace

    bootstrap_workspace(tmp_path)
    ctx = AgentContextBuilder(tmp_path)
    prompt = ctx.build_system_prompt(
        task_hint="Open https://example.com and scrape the pricing table"
    )
    assert "truncated to fit system prompt budget" not in prompt
    assert "## TOOLS.md" in prompt
    assert "## IDENTITY.md" in prompt
    assert "### Skill: web-scraping" in prompt
    assert "<name>behavior-assay-taxonomy</name>" in prompt
    assert "<name>weather</name>" in prompt


def test_context_builder_water_fills_unused_section_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from annolid.core.agent.context import AgentContextBuilder

    # Empty memory + no always skills: bootstrap should receive unused budget
    # instead of truncating to 35% of available chars.
    (tmp_path / "AGENTS.md").write_text("X" * 2400, encoding="utf-8")
    monkeypatch.setenv("ANNOLID_AGENT_SYSTEM_PROMPT_MAX_CHARS", "5000")
    ctx = AgentContextBuilder(tmp_path)
    # Use isolated empty builtin dir so skills_summary is empty and all non-identity
    # budget can flow to bootstrap cleanly.
    ctx.skills = AgentSkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
    prompt = ctx.build_system_prompt()
    assert "truncated to fit system prompt budget" not in prompt
    assert ("X" * 2400) in prompt
