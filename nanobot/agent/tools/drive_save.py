"""Tool for saving files to a local Google Drive sync folder."""

import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

# Minimal mobile-friendly HTML template for rendering markdown
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 720px;
    margin: 0 auto;
    padding: 16px;
    line-height: 1.6;
    color: #1a1a1a;
    background: #fff;
}}
h1 {{ font-size: 1.5em; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; }}
h2 {{ font-size: 1.3em; margin-top: 1.5em; }}
h3 {{ font-size: 1.1em; margin-top: 1.2em; }}
code {{
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
}}
pre {{
    background: #f4f4f4;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 0.85em;
    line-height: 1.4;
}}
pre code {{ background: none; padding: 0; }}
blockquote {{
    border-left: 3px solid #ccc;
    margin: 1em 0;
    padding: 0.5em 1em;
    color: #555;
}}
ul, ol {{ padding-left: 1.5em; }}
li {{ margin: 4px 0; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 0.9em;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}}
th {{ background: #f8f8f8; font-weight: 600; }}
a {{ color: #0066cc; }}
hr {{ border: none; border-top: 1px solid #e0e0e0; margin: 2em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _md_to_html(markdown: str) -> str:
    """
    Convert markdown to HTML using simple regex transformations.

    Not a full parser — covers headings, bold, italic, code blocks,
    inline code, links, lists, and paragraphs. Good enough for
    mobile viewing of typical documents.
    """
    html = markdown

    # Code blocks (must be first to protect content)
    code_blocks: list[str] = []

    def save_code(m: re.Match) -> str:
        lang = m.group(1) or ""
        code = m.group(2).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        code_blocks.append(f"<pre><code>{code}</code></pre>")
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    html = re.sub(r"```(\w*)\n([\s\S]*?)```", save_code, html)

    # Inline code
    inline_codes: list[str] = []

    def save_inline(m: re.Match) -> str:
        code = m.group(1).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        inline_codes.append(f"<code>{code}</code>")
        return f"\x00INLINE{len(inline_codes) - 1}\x00"

    html = re.sub(r"`([^`]+)`", save_inline, html)

    # Escape HTML
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Headings
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"__(.+?)__", r"<strong>\1</strong>", html)
    html = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<em>\1</em>", html)

    # Links
    html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)

    # Horizontal rule
    html = re.sub(r"^---+$", "<hr>", html, flags=re.MULTILINE)

    # Bullet lists (simple one-level)
    html = re.sub(r"^[\-\*] (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
    html = re.sub(r"(<li>.*?</li>\n?)+", lambda m: f"<ul>{m.group(0)}</ul>", html)

    # Numbered lists
    html = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)

    # Blockquotes
    html = re.sub(r"^&gt; (.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)

    # Paragraphs: wrap standalone text lines
    lines = html.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("<"):
            result.append(f"<p>{stripped}</p>")
        else:
            result.append(line)
    html = "\n".join(result)

    # Restore code blocks and inline code
    for i, code in enumerate(code_blocks):
        html = html.replace(f"\x00CODE{i}\x00", code)
    for i, code in enumerate(inline_codes):
        html = html.replace(f"\x00INLINE{i}\x00", code)

    return html


class SaveToDriveTool(Tool):
    """Tool to save files to a local folder synced with Google Drive."""

    def __init__(self, sync_folder: str = "", auto_html: bool = True):
        self._sync_folder = sync_folder
        self._auto_html = auto_html

    @property
    def name(self) -> str:
        return "save_to_drive"

    @property
    def description(self) -> str:
        return (
            "Save a file to the Google Drive sync folder for mobile viewing. "
            "The file will be automatically synced to Google Drive. "
            "Markdown files are auto-converted to HTML for easy mobile reading."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File name with extension (e.g. 'report.md', 'notes.txt')",
                },
                "content": {
                    "type": "string",
                    "description": "File content to save",
                },
                "subfolder": {
                    "type": "string",
                    "description": "Optional subfolder within the sync folder",
                },
            },
            "required": ["filename", "content"],
        }

    async def execute(
        self,
        filename: str,
        content: str,
        subfolder: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._sync_folder:
            return "Error: Drive sync folder not configured. Set tools.drive.syncFolder in config."

        try:
            base = Path(self._sync_folder)
            if subfolder:
                target_dir = base / subfolder
            else:
                target_dir = base

            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / filename

            # Write the file
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved to drive: {file_path}")

            result_parts = [f"Saved: {file_path}"]

            # Auto-convert .md to .html for mobile viewing
            if self._auto_html and filename.lower().endswith(".md"):
                html_name = filename.rsplit(".", 1)[0] + ".html"
                html_path = target_dir / html_name

                title = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()
                html_body = _md_to_html(content)
                html_content = _HTML_TEMPLATE.format(title=title, body=html_body)

                html_path.write_text(html_content, encoding="utf-8")
                logger.info(f"Auto-converted to HTML: {html_path}")
                result_parts.append(f"HTML version: {html_path}")

            return " | ".join(result_parts)

        except Exception as e:
            logger.error(f"Drive save error: {e}")
            return f"Error saving to drive: {str(e)}"
