import base64
import html
import io
import re
from typing import Any, Dict, List, Match, Optional, Tuple

import matplotlib.pyplot as plt
from qtpy import QtGui

try:
    from markdown_it import MarkdownIt  # type: ignore

    try:
        from markdown_it.extensions.tasklists import tasklists  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        tasklists = None  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    MarkdownIt = None  # type: ignore
    tasklists = None  # type: ignore

try:
    import markdown  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    markdown = None  # type: ignore

try:
    import bleach  # type: ignore
    from bleach.sanitizer import CSSSanitizer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    bleach = None  # type: ignore
    CSSSanitizer = None  # type: ignore


class RichTextRenderer:
    """Convert Markdown (with light LaTeX support) into styled HTML."""

    def __init__(self) -> None:
        self._list_style = "margin: 6px 0 6px 18px;"
        self._code_style = (
            "background-color:#f0f0f0; border-radius:4px; padding:2px 4px; "
            "font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
        )
        self._pre_style = (
            "background-color:#f5f5f5; border-radius:8px; padding:12px; "
            "font-family:'JetBrains Mono','Consolas','Courier New',monospace; "
            "font-size:13px; overflow-x:auto;"
        )
        self._blockquote_style = (
            "margin: 8px 0; padding: 6px 12px; border-left: 4px solid #d0d7de; "
            "color:#57606a; background-color:#f6f8fa; border-radius: 0 6px 6px 0;"
        )
        self._md = self._build_markdown_engine()
        self._css_sanitizer = (
            CSSSanitizer(
                allowed_css_properties={
                    "color",
                    "background-color",
                    "font-family",
                    "font-size",
                    "font-weight",
                    "font-style",
                    "text-align",
                    "margin",
                    "padding",
                    "border",
                    "border-radius",
                    "line-height",
                    "border-collapse",
                    "border-spacing",
                    "width",
                    "max-width",
                    "height",
                    "max-height",
                    "overflow",
                    "overflow-x",
                    "display",
                    "flex-direction",
                    "gap",
                    "align-self",
                    "justify-content",
                    "white-space",
                },
                allow_hyphenated_properties=True,
            )
            if CSSSanitizer is not None
            else None
        )

    # ------------------------------------------------------------------ public
    def render(self, text: str) -> str:
        """Render Markdown text into styled HTML."""
        if not text:
            return ""

        sanitized_text, literal_tokens = self._preprocess(text)
        sanitized_text, special_tokens = self._extract_special_tags(sanitized_text)
        processed_text, placeholders = self._extract_math_placeholders(sanitized_text)
        html_content = self._convert_markdown_to_html(
            processed_text, raw_text=sanitized_text
        )

        for token, data in placeholders.items():
            if data["block"]:
                pattern = re.compile(
                    rf"<p[^>]*>\s*{re.escape(token)}\s*</p>", re.IGNORECASE
                )
                html_content, replaced = pattern.subn(
                    lambda _match: data["html"], html_content
                )
                if not replaced:
                    html_content = html_content.replace(token, data["html"])
            else:
                html_content = html_content.replace(token, data["html"])

        html_content = self._style_rich_html(html_content)

        for token, value in literal_tokens.items():
            html_content = html_content.replace(token, value)
            html_content = html_content.replace(html.escape(token, quote=False), value)

        for token, value in special_tokens.items():
            html_content = html_content.replace(token, value)
            html_content = html_content.replace(html.escape(token, quote=False), value)

        return html_content

    # ------------------------------------------------------------ helpers
    @staticmethod
    def escape_html(text: Optional[str]) -> str:
        if text is None:
            return ""
        return html.escape(text, quote=False)

    def _build_markdown_engine(self) -> Optional["MarkdownIt"]:
        if MarkdownIt is None:
            return None

        md = MarkdownIt("commonmark", {"typographer": True, "linkify": True})
        md = md.enable("table").enable("strikethrough")
        if tasklists is not None:
            md = md.use(tasklists, clickable_checkbox=False)
        return md

    def _sanitize_rendered_html(self, html_content: str) -> str:
        if bleach is None:
            return html_content

        allowed_tags = bleach.sanitizer.ALLOWED_TAGS.union(
            {
                "p",
                "pre",
                "code",
                "img",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "table",
                "thead",
                "tbody",
                "tr",
                "th",
                "td",
                "blockquote",
            }
        )
        base_attrs = dict(bleach.sanitizer.ALLOWED_ATTRIBUTES)

        def _extend(tag: str, attrs):
            base_attrs[tag] = sorted(set(base_attrs.get(tag, []) + attrs))

        for tag in ("div", "span", "p", "code", "pre"):
            _extend(tag, ["class", "style"])
        _extend("img", ["src", "alt", "style", "class"])
        _extend("a", ["href", "title", "rel", "class", "style"])
        _extend("td", ["align", "class", "style"])
        _extend("th", ["align", "class", "style"])

        allowed_attrs = base_attrs
        return bleach.clean(
            html_content,
            tags=allowed_tags,
            attributes=allowed_attrs,
            css_sanitizer=self._css_sanitizer,
        )

    # ----------------------------------------------------------- latex helpers
    def _latex_to_image_base64(
        self, latex_string: str, *, fontsize: int, dpi: int, inline: bool
    ) -> Optional[str]:
        try:
            plt.clf()
            if inline:
                width = max(1.6, 0.085 * len(latex_string) + 0.6)
                height = 0.8
            else:
                width = max(2.8, 0.12 * len(latex_string) + 1.2)
                line_breaks = latex_string.count("\\\\") + latex_string.count("\n")
                height = max(1.2, 0.9 + 0.25 * line_breaks)

            fig = plt.figure(figsize=(width, height), dpi=dpi)
            fig.patch.set_alpha(0.0)
            fig.text(
                0.5,
                0.5,
                rf"${latex_string}$",
                fontsize=fontsize,
                ha="center",
                va="center",
            )
            plt.axis("off")

            buf = io.BytesIO()
            plt.savefig(
                buf,
                format="png",
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=True,
            )
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            return image_base64
        except Exception as exc:  # pragma: no cover - rendering failure path
            print(f"Error rendering LaTeX: {exc}")
            return None

    def _render_latex_html(self, latex_string: str, inline: bool) -> str:
        base64_data = self._latex_to_image_base64(
            latex_string,
            fontsize=14 if inline else 20,
            dpi=220,
            inline=inline,
        )
        if base64_data:
            alt_text = self.escape_html(latex_string)
            if inline:
                return (
                    '<span class="math-inline">'
                    f'<img class="math-img" alt="{alt_text}" '
                    f'src="data:image/png;base64,{base64_data}"/>'
                    "</span>"
                )
            return (
                '<div class="math-block">'
                f'<img class="math-img" alt="{alt_text}" '
                f'src="data:image/png;base64,{base64_data}"/>'
                "</div>"
            )

        return (
            '<span class="math-error">'
            f"⚠ Unable to render LaTeX: {self.escape_html(latex_string)}"
            "</span>"
        )

    def _extract_math_placeholders(
        self, text: str
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        placeholders: Dict[str, Dict[str, Any]] = {}
        counter = 0

        def store(expr: str, inline: bool) -> str:
            nonlocal counter
            token = f"__MATH_{counter}__"
            counter += 1
            placeholders[token] = {
                "html": self._render_latex_html(expr, inline=inline),
                "block": not inline,
            }
            return token

        def replace_block(match: Match[str]) -> str:
            return store(match.group(1).strip(), inline=False)

        def replace_inline(match: Match[str]) -> str:
            return store(match.group(1).strip(), inline=True)

        text = re.sub(
            r"(?<!\\)\$\$(.+?)(?<!\\)\$\$",
            replace_block,
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"(?<!\\)\\\[(.+?)(?<!\\)\\\]",
            replace_block,
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$",
            replace_inline,
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"(?<!\\)\\\((.+?)(?<!\\)\\\)",
            replace_inline,
            text,
            flags=re.DOTALL,
        )
        return text, placeholders

    # ------------------------------------------------------------ markdown core
    def _basic_markdown_to_html(self, text: str) -> str:
        def escape_segment(segment: str) -> str:
            return html.escape(segment, quote=False)

        token_map: Dict[str, str] = {}

        def add_token(snippet: str, kind: str) -> str:
            token = f"<<ANNOLID{kind}{len(token_map)}>>"
            token_map[token] = snippet
            return token

        def replace_code(match: Match[str]) -> str:
            return add_token(
                f"<code>{escape_segment(match.group(1))}</code>",
                "CODE",
            )

        protected = re.sub(r"`([^`]+)`", replace_code, text)

        def sanitize_url(url: str) -> str:
            url = url.strip()
            if url.startswith("http://") or url.startswith("https://"):
                return html.escape(url, quote=True)
            return "#"

        def replace_image(match: Match[str]) -> str:
            alt = escape_segment(match.group(1))
            url = sanitize_url(match.group(2))
            return add_token(f'<img alt="{alt}" src="{url}"/>', "IMG")

        protected = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, protected)

        def replace_bold(match: Match[str]) -> str:
            return add_token(
                f"<strong>{escape_segment(match.group(1))}</strong>",
                "BOLD",
            )

        def replace_strong_em(match: Match[str]) -> str:
            return add_token(
                f"<strong><em>{escape_segment(match.group(1))}</em></strong>",
                "SMEM",
            )

        protected = re.sub(r"\*\*\*(.+?)\*\*\*", replace_strong_em, protected)
        protected = re.sub(r"___(.+?)___", replace_strong_em, protected)
        protected = re.sub(r"\*\*(.+?)\*\*", replace_bold, protected)
        protected = re.sub(r"__(.+?)__", replace_bold, protected)

        def replace_italic(match: Match[str]) -> str:
            return add_token(
                f"<em>{escape_segment(match.group(1))}</em>",
                "EM",
            )

        protected = re.sub(
            r"(?<!\*)\*(?!\*)([^\n*][\s\S]*?[^\n*])\*(?!\*)",
            replace_italic,
            protected,
        )
        protected = re.sub(
            r"(?<!_)_(?!_)([^\n_][\s\S]*?[^\n_])_(?!_)",
            replace_italic,
            protected,
        )

        protected = re.sub(
            r"~~(.+?)~~",
            lambda m: f"<del>{escape_segment(m.group(1))}</del>",
            protected,
        )

        def replace_link(match: Match[str]) -> str:
            label = escape_segment(match.group(1))
            url = sanitize_url(match.group(2))
            return f'<a href="{url}">{label}</a>'

        protected = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, protected)

        protected = re.sub(
            r"(?P<url>https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)",
            lambda m: f'<a href="{sanitize_url(m.group("url"))}">'
            f"{escape_segment(m.group('url'))}</a>",
            protected,
        )

        def replace_heading(match: Match[str]) -> str:
            level = len(match.group(1))
            content = escape_segment(match.group(2).strip())
            return f"<h{level}>{content}</h{level}>"

        protected = re.sub(
            r"^(#{1,6})\s+(.+)$", replace_heading, protected, flags=re.MULTILINE
        )

        blocks: List[str] = []
        buf: List[str] = []
        list_items: List[str] = []
        list_type: Optional[str] = None

        def flush_buf() -> None:
            if not buf:
                return
            paragraph = "<br/>".join(escape_segment(x) for x in buf)
            blocks.append(f"<p>{paragraph}</p>")
            buf.clear()

        def flush_list() -> None:
            nonlocal list_type
            if not list_items:
                list_type = None
                return
            tag = "ol" if list_type == "ordered" else "ul"
            blocks.append(f"<{tag}>" + "".join(list_items) + f"</{tag}>")
            list_items.clear()
            list_type = None

        for line in protected.splitlines():
            stripped = line.strip()
            if not stripped:
                flush_buf()
                flush_list()
                continue

            if stripped.startswith(">"):
                flush_buf()
                flush_list()
                content = stripped[1:].lstrip()
                blocks.append(f"<blockquote>{escape_segment(content)}</blockquote>")
                continue

            unordered = re.match(r"^[-*+]\s+(.*)$", stripped)
            ordered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
            if unordered or ordered:
                flush_buf()
                desired_type = "ordered" if ordered else "unordered"
                if list_type and list_type != desired_type:
                    flush_list()
                if not list_type:
                    list_type = desired_type
                content = unordered.group(1) if unordered else ordered.group(1)
                list_items.append(f"<li>{escape_segment(content)}</li>")
                continue

            if list_type:
                flush_list()

            buf.append(line)

        flush_buf()
        flush_list()

        html_out = "".join(blocks) if blocks else escape_segment(protected)

        for token, html_snippet in token_map.items():
            html_out = html_out.replace(html.escape(token, quote=False), html_snippet)
            html_out = html_out.replace(token, html_snippet)

        return html_out

    def _convert_markdown_to_html(self, text: str, raw_text: str) -> str:
        if self._md is not None:
            try:
                rendered = self._md.render(text)
                if rendered:
                    return self._sanitize_rendered_html(rendered)
            except Exception:
                pass

        doc_cls = getattr(QtGui, "QTextDocument", None)
        if doc_cls is not None:
            document = doc_cls()
            set_markdown = getattr(document, "setMarkdown", None)
            if callable(set_markdown):
                try:
                    set_markdown(text)
                    html_content = document.toHtml()
                    if html_content:
                        return html_content
                except Exception:
                    pass

            fragment_ctor = getattr(QtGui.QTextDocumentFragment, "fromMarkdown", None)
            if fragment_ctor is not None:
                try:
                    fragment = fragment_ctor(text)
                    html_content = fragment.toHtml()
                    if html_content:
                        return html_content
                except Exception:
                    pass

        if markdown is not None:
            try:
                rendered = markdown.markdown(
                    text,
                    extensions=["extra", "sane_lists", "tables", "fenced_code"],
                    output_format="html5",
                )
                if rendered:
                    return self._sanitize_rendered_html(rendered)
            except Exception:
                pass

        return self._basic_markdown_to_html(raw_text)

    # ------------------------------------------------------------- styling/pre
    def _style_rich_html(self, html_content: str) -> str:
        html_content = re.sub(
            r"<pre(?![^>]*style=)([^>]*)>",
            lambda match: f'<pre{match.group(1)} style="{self._pre_style}">',
            html_content,
        )
        html_content = re.sub(
            r"<code(?![^>]*style=)([^>]*)>",
            lambda match: f'<code{match.group(1)} style="{self._code_style}">',
            html_content,
        )
        html_content = re.sub(
            r"<blockquote(?![^>]*style=)([^>]*)>",
            lambda match: f'<blockquote{match.group(1)} style="{self._blockquote_style}">',
            html_content,
        )
        html_content = re.sub(
            r"<(ul|ol)(?![^>]*style=)([^>]*)>",
            lambda match: f'<{match.group(1)}{match.group(2)} style="{self._list_style}">',
            html_content,
        )
        return html_content

    # ------------------------------------------------------------- preprocess
    def _preprocess(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return "", {}

        replacements: Dict[str, str] = {}

        def replace_parenthesized_star(match: Match[str]) -> str:
            spacing = match.group(1) or ""
            return f"({spacing}•"

        sanitized = re.sub(
            r"(?<!\\)\((\s*)\*(?!\*)",
            replace_parenthesized_star,
            text,
        )

        sanitized = re.sub(
            r"(?<=\()\*(?=\s*\*\*)",
            "•",
            sanitized,
        )

        if "```" not in sanitized and "~~~" not in sanitized:
            sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

        stripped = sanitized.strip()
        multiline_group = re.match(r"^\(\s*(.+?)\s*\)\s*$", stripped, flags=re.DOTALL)
        if multiline_group and "\n" in multiline_group.group(1):
            sanitized = multiline_group.group(1).strip()

        return sanitized, replacements

    def _extract_special_tags(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return "", {}
        # Drop non-HTML tags leaked by some models (e.g., <think> ... </think>)
        text = re.sub(r"</?think[^>]*>", "", text, flags=re.IGNORECASE)
        return text, {}
