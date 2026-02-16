"""Text-to-speech provider using Aliyun DashScope."""

import hashlib
import os
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


class DashScopeTTSProvider:
    """
    TTS provider using Aliyun DashScope CosyVoice API.

    Calls the DashScope text2audio endpoint directly (no Java backend needed).
    Caches synthesized audio to avoid redundant API calls.
    """

    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2audio/generation"
    MAX_TEXT_LENGTH = 3000

    # Available voice IDs (CosyVoice)
    VOICES = {
        "Cherry": "Sweet, lively female (default)",
        "Serena": "Gentle, warm female",
        "Ethan": "Warm, sunny male",
        "Nini": "Friendly, approachable female",
        "Maia": "Intellectual, warm female",
        "Moon": "Solid, magnetic male",
    }

    def __init__(self, api_key: str | None = None, default_voice: str = "Cherry"):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.default_voice = default_voice
        self.cache_dir = Path.home() / ".nanobot" / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, text: str, voice: str) -> str:
        """Generate a cache filename from text + voice."""
        h = hashlib.sha256(f"{text}:{voice}".encode()).hexdigest()[:16]
        return h

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Synthesize text to speech.

        Args:
            text: Text to convert (max 3000 chars).
            voice: CosyVoice voice ID (default: Cherry).

        Returns:
            Path to the synthesized MP3 file, or empty string on failure.
        """
        if not self.api_key:
            logger.error("DashScope API key not configured for TTS")
            return ""

        voice = voice or self.default_voice

        # Truncate if too long
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(f"TTS text truncated from {len(text)} to {self.MAX_TEXT_LENGTH} chars")
            text = text[: self.MAX_TEXT_LENGTH]

        # Check cache
        key = self._cache_key(text, voice)
        cache_file = self.cache_dir / f"{key}.mp3"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            logger.debug(f"TTS cache hit: {cache_file}")
            return str(cache_file)

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": "cosyvoice-v1",
                    "input": {
                        "text": text,
                        "voice": voice,
                    },
                }

                response = await client.post(
                    self.API_URL,
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )

                content_type = response.headers.get("content-type", "")

                if "json" in content_type:
                    # JSON response — may contain audio_url or error
                    data = response.json()
                    if response.status_code != 200:
                        msg = data.get("message", data.get("error", "unknown error"))
                        logger.error(f"DashScope TTS API error: {msg}")
                        return ""

                    # Extract audio URL from response
                    output = data.get("output", {})
                    audio_url = (
                        output.get("audio_url")
                        or output.get("audio", {}).get("url")
                    )

                    if not audio_url:
                        logger.error(f"No audio URL in DashScope response: {data}")
                        return ""

                    # Download audio
                    audio_resp = await client.get(audio_url, timeout=30.0)
                    audio_resp.raise_for_status()
                    audio_bytes = audio_resp.content
                else:
                    # Binary audio response
                    response.raise_for_status()
                    audio_bytes = response.content

                if not audio_bytes:
                    logger.error("DashScope TTS returned empty audio")
                    return ""

                cache_file.write_bytes(audio_bytes)
                logger.info(f"TTS synthesized ({len(audio_bytes)} bytes, voice={voice}): {cache_file}")
                return str(cache_file)

        except Exception as e:
            logger.error(f"DashScope TTS error: {e}")
            return ""
