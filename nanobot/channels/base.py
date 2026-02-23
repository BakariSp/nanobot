"""Base channel interface for chat platforms."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.

    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the nanobot message bus.
    """

    name: str = "base"

    def __init__(self, config: Any, bus: MessageBus):
        """
        Initialize the channel.

        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
        """
        self.config = config
        self.bus = bus
        self._running = False
        self._known_users: set[str] = set()
        self._user_cap_file = Path.home() / ".nanobot" / f"user_cap_{self.name}.json"
        self._load_known_users()
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.
        
        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass
    
    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.
        
        Args:
            msg: The message to send.
        """
        pass
    
    def _load_known_users(self) -> None:
        """Load known users from disk (survives restarts)."""
        if self._user_cap_file.exists():
            try:
                data = json.loads(self._user_cap_file.read_text(encoding="utf-8"))
                self._known_users = set(data.get("users", []))
            except Exception:
                pass

    def _save_known_users(self) -> None:
        """Persist known users to disk."""
        self._user_cap_file.parent.mkdir(parents=True, exist_ok=True)
        self._user_cap_file.write_text(
            json.dumps({"users": sorted(self._known_users)}),
            encoding="utf-8",
        )

    def is_allowed(self, sender_id: str) -> bool:
        """
        Check if a sender is allowed to use this bot.

        Args:
            sender_id: The sender's identifier.

        Returns:
            True if allowed, False otherwise.
        """
        allow_list = getattr(self.config, "allow_from", [])

        # If no allow list, allow everyone
        if not allow_list:
            return True

        sender_str = str(sender_id)
        if sender_str in allow_list:
            return True
        if "|" in sender_str:
            for part in sender_str.split("|"):
                if part and part in allow_list:
                    return True
        return False

    def check_user_cap(self, sender_id: str) -> bool:
        """Check if the user is within the max_users cap.

        Returns True if the user is admitted (existing or newly added),
        False if the cap has been reached and this is a new user.
        Always returns True when max_users is 0 (unlimited).
        """
        max_users = getattr(self.config, "max_users", 0)
        if max_users <= 0:
            return True

        primary_id = str(sender_id).split("|")[0]
        if primary_id in self._known_users:
            return True

        if len(self._known_users) >= max_users:
            return False

        self._known_users.add(primary_id)
        self._save_known_users()
        logger.info(
            f"New test user admitted: {primary_id} "
            f"({len(self._known_users)}/{max_users})"
        )
        return True

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """
        Handle an incoming message from the chat platform.

        This method checks permissions and forwards to the bus.

        Returns:
            None on success, or a rejection reason string
            ("not_allowed" | "user_cap_reached").
        """
        if not self.is_allowed(sender_id):
            logger.warning(
                f"Access denied for sender {sender_id} on channel {self.name}. "
                f"Add them to allowFrom list in config to grant access."
            )
            return "not_allowed"

        if not self.check_user_cap(sender_id):
            max_users = getattr(self.config, "max_users", 0)
            logger.warning(
                f"User cap reached ({len(self._known_users)}/{max_users}), "
                f"rejecting new user {sender_id} on {self.name}"
            )
            return "user_cap_reached"

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {}
        )

        await self.bus.publish_inbound(msg)
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running
