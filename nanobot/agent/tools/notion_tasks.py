"""Tool for reading and managing tasks in a Notion database."""

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


def _notion_headers(api_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }


def _extract_title(props: dict) -> str:
    for v in props.values():
        if v.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in v.get("title", []))
    return "(untitled)"


def _extract_status(props: dict) -> str:
    for key in ("Status", "status"):
        if key in props and props[key].get("type") == "status":
            obj = props[key].get("status")
            if obj:
                return obj.get("name", "")
    return ""


def _extract_checkbox(props: dict) -> bool:
    for key in ("Done", "done", "Completed", "completed"):
        if key in props and props[key].get("type") == "checkbox":
            return props[key].get("checkbox", False)
    return False


class NotionTasksTool(Tool):
    """Read and manage tasks in a Notion database."""

    def __init__(self, api_token: str = "", database_id: str = ""):
        self._api_token = api_token
        self._database_id = database_id

    @property
    def name(self) -> str:
        return "notion_tasks"

    @property
    def description(self) -> str:
        return (
            "Read and manage tasks in Notion. Actions:\n"
            "- 'list': Query tasks from the database (optional filter_status)\n"
            "- 'get': Read a single task by page_id\n"
            "- 'update': Change a task's done (checkbox) or status property\n"
            "Use this to check pending tasks, mark completion, or sync progress."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "get", "update"],
                    "description": "Action to perform",
                },
                "database_id": {
                    "type": "string",
                    "description": "Notion database ID (uses default if omitted)",
                },
                "page_id": {
                    "type": "string",
                    "description": "Page ID (required for 'get' and 'update')",
                },
                "done": {
                    "type": "boolean",
                    "description": "Set Done checkbox value (for 'update')",
                },
                "status": {
                    "type": "string",
                    "description": "Set Status property value, e.g. 'Done', 'In progress' (for 'update')",
                },
                "filter_status": {
                    "type": "string",
                    "description": "Filter tasks by Status value (for 'list')",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        if not self._api_token:
            return "Error: Notion API token not configured."

        db_id = kwargs.get("database_id") or self._database_id

        if action == "list":
            return await self._list_tasks(db_id, kwargs.get("filter_status"))
        elif action == "get":
            page_id = kwargs.get("page_id")
            if not page_id:
                return "Error: page_id is required for 'get'."
            return await self._get_task(page_id)
        elif action == "update":
            page_id = kwargs.get("page_id")
            if not page_id:
                return "Error: page_id is required for 'update'."
            return await self._update_task(page_id, kwargs.get("done"), kwargs.get("status"))
        else:
            return f"Unknown action: {action}. Use 'list', 'get', or 'update'."

    async def _list_tasks(self, database_id: str, filter_status: str | None = None) -> str:
        if not database_id:
            return "Error: No database_id provided and no default configured."
        try:
            import httpx

            headers = _notion_headers(self._api_token)
            body: dict[str, Any] = {"page_size": 100}

            if filter_status:
                body["filter"] = {
                    "property": "Status",
                    "status": {"equals": filter_status},
                }

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://api.notion.com/v1/databases/{database_id}/query",
                    json=body,
                    headers=headers,
                    timeout=30.0,
                )
                if resp.status_code >= 400:
                    return f"Error querying Notion: {resp.text}"

                results = resp.json().get("results", [])

                if not results:
                    return "No tasks found."

                lines = []
                for page in results:
                    props = page.get("properties", {})
                    title = _extract_title(props)
                    status = _extract_status(props)
                    done = _extract_checkbox(props)
                    page_id = page["id"]
                    url = page.get("url", "")

                    mark = "[x]" if done else "[ ]"
                    status_str = f" ({status})" if status else ""
                    lines.append(f"{mark} {title}{status_str}\n    id: {page_id}\n    {url}")

                return f"Found {len(results)} tasks:\n\n" + "\n\n".join(lines)

        except Exception as e:
            logger.error(f"Notion list tasks error: {e}")
            return f"Error: {e}"

    async def _get_task(self, page_id: str) -> str:
        try:
            import httpx

            headers = _notion_headers(self._api_token)

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://api.notion.com/v1/pages/{page_id}",
                    headers=headers,
                    timeout=30.0,
                )
                if resp.status_code >= 400:
                    return f"Error: {resp.text}"

                page = resp.json()
                props = page.get("properties", {})
                title = _extract_title(props)
                status = _extract_status(props)
                done = _extract_checkbox(props)
                url = page.get("url", "")

                # Fetch page content blocks
                blocks_resp = await client.get(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    headers=headers,
                    timeout=30.0,
                )
                content_preview = ""
                if blocks_resp.status_code < 400:
                    blocks = blocks_resp.json().get("results", [])
                    texts = []
                    for b in blocks[:20]:
                        btype = b.get("type", "")
                        block_data = b.get(btype, {})
                        rich_texts = block_data.get("rich_text", [])
                        text = "".join(rt.get("plain_text", "") for rt in rich_texts)
                        if text:
                            texts.append(text)
                    content_preview = "\n".join(texts)

                mark = "[x]" if done else "[ ]"
                parts = [
                    f"{mark} {title}",
                    f"Status: {status or 'N/A'}",
                    f"URL: {url}",
                ]
                if content_preview:
                    parts.append(f"\nContent:\n{content_preview}")

                return "\n".join(parts)

        except Exception as e:
            logger.error(f"Notion get task error: {e}")
            return f"Error: {e}"

    async def _update_task(self, page_id: str, done: bool | None, status: str | None) -> str:
        if done is None and status is None:
            return "Error: Provide 'done' (bool) or 'status' (string) to update."
        try:
            import httpx

            headers = _notion_headers(self._api_token)
            properties: dict[str, Any] = {}

            if done is not None:
                properties["Done"] = {"checkbox": done}
            if status is not None:
                properties["Status"] = {"status": {"name": status}}

            async with httpx.AsyncClient() as client:
                resp = await client.patch(
                    f"https://api.notion.com/v1/pages/{page_id}",
                    json={"properties": properties},
                    headers=headers,
                    timeout=30.0,
                )
                if resp.status_code >= 400:
                    return f"Error updating task: {resp.text}"

                page = resp.json()
                props = page.get("properties", {})
                title = _extract_title(props)
                url = page.get("url", "")

                return f"Updated: {title}\n{url}"

        except Exception as e:
            logger.error(f"Notion update task error: {e}")
            return f"Error: {e}"
