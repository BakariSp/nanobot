"""Tool for saving content to Notion."""

import re
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


def _markdown_to_notion_blocks(content: str) -> list[dict[str, Any]]:
    """
    Convert markdown text to Notion block objects.

    Handles: headings, paragraphs, code blocks, bullet lists.
    """
    blocks: list[dict[str, Any]] = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith("```"):
            lang_match = re.match(r"```(\w*)", line.strip())
            lang = lang_match.group(1) if lang_match else "plain text"
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = "\n".join(code_lines)
            if code_text:
                blocks.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": lang or "plain text",
                        "rich_text": [{"type": "text", "text": {"content": code_text[:2000]}}],
                    },
                })
            continue

        # Heading 1
        if line.startswith("# ") and not line.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}],
                },
            })
            i += 1
            continue

        # Heading 2
        if line.startswith("## ") and not line.startswith("### "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}],
                },
            })
            i += 1
            continue

        # Heading 3
        if line.startswith("### "):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": line[4:].strip()}}],
                },
            })
            i += 1
            continue

        # Bullet list
        bullet_match = re.match(r"^[\-\*]\s+(.+)$", line)
        if bullet_match:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": bullet_match.group(1)}}],
                },
            })
            i += 1
            continue

        # Numbered list
        num_match = re.match(r"^\d+\.\s+(.+)$", line)
        if num_match:
            blocks.append({
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": num_match.group(1)}}],
                },
            })
            i += 1
            continue

        # Empty line — skip
        if not line.strip():
            i += 1
            continue

        # Paragraph (collect consecutive non-empty, non-special lines)
        para_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if (
                not next_line.strip()
                or next_line.startswith("#")
                or next_line.strip().startswith("```")
                or re.match(r"^[\-\*]\s+", next_line)
                or re.match(r"^\d+\.\s+", next_line)
            ):
                break
            para_lines.append(next_line)
            i += 1

        text = "\n".join(para_lines).strip()
        if text:
            # Notion has a 2000-char limit per rich_text element
            for chunk_start in range(0, len(text), 2000):
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"type": "text", "text": {"content": text[chunk_start:chunk_start + 2000]}}
                        ],
                    },
                })

    return blocks


async def create_notion_page(
    api_token: str,
    database_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> str | None:
    """Create a page in a Notion database.

    Returns:
        Page URL on success, None on failure.
    """
    try:
        import httpx

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

        properties: dict[str, Any] = {
            "title": {
                "title": [{"type": "text", "text": {"content": title}}],
            },
        }
        if tags:
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in tags],
            }

        blocks = _markdown_to_notion_blocks(content)
        payload: dict[str, Any] = {
            "parent": {"database_id": database_id},
            "properties": properties,
            "children": blocks[:100],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.notion.com/v1/pages",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            if response.status_code >= 400:
                error_data = response.json()
                msg = error_data.get("message", str(error_data))
                # If Tags property doesn't exist, retry without it
                if "is not a property that exists" in msg and "Tags" in properties:
                    logger.warning(f"Notion database has no Tags property, retrying without tags")
                    del properties["Tags"]
                    payload["properties"] = properties
                    response = await client.post(
                        "https://api.notion.com/v1/pages",
                        json=payload,
                        headers=headers,
                        timeout=30.0,
                    )
                    if response.status_code >= 400:
                        error_data = response.json()
                        msg = error_data.get("message", str(error_data))
                        logger.error(f"Notion page creation failed (retry): {msg}")
                        return None
                else:
                    logger.error(f"Notion page creation failed: {msg}")
                    return None

            data = response.json()
            page_id = data.get("id", "")
            page_url = data.get("url", "")

            remaining = blocks[100:]
            while remaining:
                batch = remaining[:100]
                remaining = remaining[100:]
                await client.patch(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    json={"children": batch},
                    headers=headers,
                    timeout=30.0,
                )

            logger.info(f"Created Notion page: {page_url or page_id}")
            return page_url or None

    except ImportError:
        logger.error("httpx not installed, cannot push to Notion")
        return None
    except Exception as e:
        logger.error(f"Notion page creation error: {e}")
        return None


class SaveToNotionTool(Tool):
    """Tool to save content to a Notion database as a new page."""

    def __init__(self, api_token: str = "", database_id: str = ""):
        self._api_token = api_token
        self._database_id = database_id

    @property
    def name(self) -> str:
        return "save_to_notion"

    @property
    def description(self) -> str:
        return (
            "Save a document to Notion. Creates a new page in the configured Notion database. "
            "Content should be professional, third-person, without emoji or conversation artifacts. "
            "Use this for key archival documents that need structured storage."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Page title (clear, professional)",
                },
                "content": {
                    "type": "string",
                    "description": "Page content in markdown format",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization",
                },
            },
            "required": ["title", "content"],
        }

    async def execute(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._api_token:
            return "Error: Notion API token not configured. Set tools.notion.apiToken in config."
        if not self._database_id:
            return "Error: Notion database ID not configured. Set tools.notion.databaseId in config."

        url = await create_notion_page(
            api_token=self._api_token,
            database_id=self._database_id,
            title=title,
            content=content,
            tags=tags,
        )
        if url:
            return f"Saved to Notion: {url}"
        return "Error saving to Notion (check logs for details)"
