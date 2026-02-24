"""Tool for sending voice messages via TTS."""

from typing import Any, Callable, Awaitable, Protocol

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class TTSProvider(Protocol):
    """Any TTS provider that has an async synthesize method."""
    async def synthesize(self, text: str, voice: str | None = None, **kwargs: Any) -> str: ...


class VoiceReplyTool(Tool):
    """
    Tool to send a voice message by converting text to speech.

    Supports pluggable TTS providers (Volcengine, DashScope, etc.)
    as long as they implement ``synthesize(text, voice) -> path``.
    """

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        tts_provider: TTSProvider | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._tts = tts_provider
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    @property
    def name(self) -> str:
        return "voice_reply"

    @property
    def description(self) -> str:
        return (
            "Send a voice message to the user by converting text to speech. "
            "Use ONLY for short, casual replies (greetings, check-ins, encouragement). "
            "Do NOT use for task responses, code, lists, or anything longer than 2 sentences. "
            "IMPORTANT: content must be plain spoken text only. No parenthetical stage directions "
            "like (笑), (停顿), (语音), (whisper) — TTS will read them out loud. "
            "Choose voice and emotion to match the mood of your message."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Short text to convert to speech (max ~200 chars recommended)",
                },
                "voice": {
                    "type": "string",
                    "description": (
                        "Speaker voice ID. Options: "
                        "zh_female_vv_uranus_bigtts (vivi, sweet female, DEFAULT), "
                        "zh_male_dayi_saturn_bigtts (大壹, warm male), "
                        "zh_male_ruyayichen_saturn_bigtts (儒雅逸辰, elegant male), "
                        "zh_female_meilinvyou_saturn_bigtts (魅力女友, charming female), "
                        "zh_female_santongyongns_saturn_bigtts (流畅女声, fluent female), "
                        "saturn_zh_female_cancan_tob (知性灿灿, intellectual female), "
                        "saturn_zh_female_keainvsheng_tob (可爱女生, cute girl), "
                        "saturn_zh_female_tiaopigongzhu_tob (调皮公主, playful princess)"
                    ),
                },
                "emotion": {
                    "type": "string",
                    "description": (
                        "Emotion/style for the voice. Options: "
                        "happy, pleased, sad, angry, annoyed, sorry, scare, surprise, tear, "
                        "narrator, comfort, chat, energetic, charming, "
                        "lovey-dovey, conniving, tsundere, storytelling, radio, serious"
                    ),
                },
                "emotion_scale": {
                    "type": "integer",
                    "description": "Emotion intensity 1-10 (default 4, higher = stronger emotion)",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        voice: str | None = None,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        **kwargs: Any,
    ) -> str:
        channel = self._default_channel
        chat_id = self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        if not self._tts:
            return "Error: TTS provider not configured. Set tools.tts in config."

        try:
            # Synthesize speech
            audio_path = await self._tts.synthesize(
                content, voice=voice, emotion=emotion, emotion_scale=emotion_scale,
            )
            if not audio_path:
                return "Error: TTS synthesis failed — check TTS provider config and logs"

            # Send voice message with text as caption
            msg = OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=content,
                media=[audio_path],
                metadata={"type": "voice"},
            )

            await self._send_callback(msg)
            logger.info(f"Voice reply sent to {channel}:{chat_id}")
            return f"Voice message sent to {channel}:{chat_id}"

        except Exception as e:
            logger.error(f"Voice reply error: {e}")
            return f"Error sending voice message: {str(e)}"
