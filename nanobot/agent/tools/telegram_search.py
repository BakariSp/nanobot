"""Telegram chat history search via Telethon (MTProto Client API).

This tool lets the agent search past Telegram messages when local memory
(MEMORY.md / HISTORY.md) is insufficient — a "last resort" retrieval layer.

Setup (one-time):
    1. Get api_id + api_hash from https://my.telegram.org
    2. Add to ~/.nanobot/config.json:
       { "tools": { "telegram_search": { "api_id": 12345, "api_hash": "abc..." } } }
    3. Run: python -m nanobot.agent.tools.telegram_search --login
       (interactive: phone → code → optional 2FA password)
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool

# Lazy-loaded to avoid import cost when tool is unused
_client = None
_client_lock = asyncio.Lock()


async def _get_client(api_id: int, api_hash: str, session_path: str, proxy: str | None = None):
    """Get or create a connected Telethon client (singleton)."""
    global _client
    async with _client_lock:
        if _client is not None and _client.is_connected():
            return _client

        from telethon import TelegramClient

        proxy_kwargs = {}
        if proxy:
            # Parse socks5://host:port
            from urllib.parse import urlparse
            p = urlparse(proxy)
            import socks
            proxy_kwargs["proxy"] = (socks.SOCKS5, p.hostname, p.port)

        _client = TelegramClient(session_path, api_id, api_hash, **proxy_kwargs)
        await _client.connect()

        if not await _client.is_user_authorized():
            raise RuntimeError(
                "Telegram session not authorized. "
                "Run: python -m nanobot.agent.tools.telegram_search --login"
            )
        return _client


def _format_message(msg) -> dict:
    """Format a Telethon Message into a compact dict."""
    return {
        "id": msg.id,
        "date": msg.date.strftime("%Y-%m-%d %H:%M") if msg.date else None,
        "from": getattr(msg.sender, "first_name", None) or str(msg.sender_id),
        "text": (msg.text or "")[:2000],  # Cap very long messages
    }


class TelegramSearchTool(Tool):
    """Search Telegram chat history for past messages.

    Use this tool as a LAST RESORT when you cannot find the information in
    your memory (MEMORY.md) or conversation history (HISTORY.md).
    """

    name = "telegram_search"
    description = (
        "Search Telegram chat history by keyword. "
        "Use ONLY when local memory and history don't have the answer. "
        "Returns matching messages with sender, date, and text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keyword or phrase to find in chat messages",
            },
            "chat": {
                "type": "string",
                "description": (
                    "Chat to search in: username (@user), phone (+8613800138000), "
                    "group title, or 'me' for Saved Messages. "
                    "Omit to search across all dialogs."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max messages to return (1-50)",
                "minimum": 1,
                "maximum": 50,
            },
            "from_user": {
                "type": "string",
                "description": "Only show messages from this user (username or name)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_id: int, api_hash: str, session_dir: str,
                 session_name: str = "nanobot_tg", proxy: str | None = None,
                 max_results: int = 20):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_path = str(Path(session_dir) / session_name)
        self.proxy = proxy
        self.max_results = max_results

    async def execute(self, query: str, chat: str | None = None,
                      limit: int | None = None, from_user: str | None = None,
                      **kwargs: Any) -> str:
        if not self.api_id or not self.api_hash:
            return "Error: telegram_search not configured (missing api_id / api_hash)"

        try:
            client = await _get_client(
                self.api_id, self.api_hash, self.session_path, self.proxy
            )
        except Exception as e:
            return f"Error connecting to Telegram: {e}"

        n = min(max(limit or self.max_results, 1), 50)

        try:
            # Resolve target chat
            entity = None
            chat_label = "all dialogs"
            if chat:
                if chat.lower() == "me":
                    entity = "me"
                    chat_label = "Saved Messages"
                else:
                    entity = await client.get_entity(chat)
                    chat_label = getattr(entity, "title", None) or getattr(entity, "first_name", chat)

            # Resolve from_user filter
            from_entity = None
            if from_user:
                from_entity = await client.get_entity(from_user)

            # Search
            if entity:
                messages = await client.get_messages(
                    entity, search=query, limit=n, from_user=from_entity
                )
            else:
                # Search across all dialogs (Telethon searches in GlobalSearch)
                from telethon.tl.functions.messages import SearchGlobalRequest
                from telethon.tl.types import InputMessagesFilterEmpty

                result = await client(SearchGlobalRequest(
                    q=query,
                    filter=InputMessagesFilterEmpty(),
                    min_date=None,
                    max_date=None,
                    offset_rate=0,
                    offset_peer=await client.get_input_entity("me"),
                    offset_id=0,
                    limit=n,
                ))
                messages = result.messages

            if not messages:
                return f"No messages found for '{query}' in {chat_label}"

            # Format results
            results = []
            for msg in messages:
                # Ensure sender is loaded
                if msg.sender is None and msg.sender_id:
                    try:
                        await msg.get_sender()
                    except Exception:
                        pass
                results.append(_format_message(msg))

            header = f"Found {len(results)} message(s) for '{query}' in {chat_label}:\n"
            lines = [header]
            for r in results:
                lines.append(f"[{r['date']}] {r['from']}: {r['text']}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error searching Telegram: {e}"


# --- CLI login helper ---
async def _interactive_login(api_id: int, api_hash: str, session_path: str, proxy: str | None = None):
    """Interactive login: phone → code → optional 2FA."""
    from telethon import TelegramClient

    proxy_kwargs = {}
    if proxy:
        from urllib.parse import urlparse
        p = urlparse(proxy)
        import socks
        proxy_kwargs["proxy"] = (socks.SOCKS5, p.hostname, p.port)

    client = TelegramClient(session_path, api_id, api_hash, **proxy_kwargs)
    await client.start()
    me = await client.get_me()
    print(f"Logged in as: {me.first_name} ({me.phone})")
    print(f"Session saved to: {session_path}.session")
    await client.disconnect()


if __name__ == "__main__":
    if "--login" not in sys.argv:
        print("Usage: python -m nanobot.agent.tools.telegram_search --login")
        print("  Interactively logs in and saves the Telethon session file.")
        sys.exit(0)

    # Load config
    config_path = Path.home() / ".nanobot" / "config.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Add telegram_search config first. See module docstring.")
        sys.exit(1)

    cfg = json.loads(config_path.read_text())
    tg = cfg.get("tools", {}).get("telegram_search", {})
    api_id = tg.get("api_id", 0)
    api_hash = tg.get("api_hash", "")
    session_name = tg.get("session_name", "nanobot_tg")
    proxy = tg.get("proxy")

    if not api_id or not api_hash:
        print("Error: tools.telegram_search.api_id and api_hash must be set in config.json")
        sys.exit(1)

    session_path = str(Path.home() / ".nanobot" / session_name)
    asyncio.run(_interactive_login(api_id, api_hash, session_path, proxy))
