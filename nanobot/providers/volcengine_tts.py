"""Text-to-speech provider using Volcengine (火山引擎) Speech API.

HTTP one-shot synthesis: send text, get base64-encoded audio back.
Docs: https://www.volcengine.com/docs/6561/97465

Required config:
  tools.tts.volcengine_app_id     — from 火山引擎语音技术控制台
  tools.tts.volcengine_token      — Bearer access token
  tools.tts.default_voice         — e.g. "ICL_zh_female_qiuling_v1_tob"
"""

import base64
import hashlib
import uuid
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


class VolcengineTTSProvider:
    """TTS via Volcengine Speech HTTP API (one-shot, non-streaming)."""

    API_URL = "https://openspeech.bytedance.com/api/v1/tts"
    MAX_TEXT_LENGTH = 3000

    def __init__(
        self,
        app_id: str,
        token: str,
        default_voice: str = "ICL_zh_female_qiuling_v1_tob",
        cluster: str = "volcano_icl",
        encoding: str = "mp3",
    ):
        self.app_id = app_id
        self.token = token
        self.default_voice = default_voice
        self.cluster = cluster
        self.encoding = encoding
        self.cache_dir = Path.home() / ".nanobot" / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, text: str, voice: str) -> str:
        h = hashlib.sha256(f"{text}:{voice}".encode()).hexdigest()[:16]
        return h

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed_ratio: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Synthesize text to speech.

        Args:
            text: Text to convert (max 3000 chars).
            voice: Volcengine voice_type ID.
            speed_ratio: Speech speed (0.2–3.0, default 1.0).

        Returns:
            Path to the synthesized audio file, or empty string on failure.
        """
        if not self.app_id or not self.token:
            logger.error("Volcengine TTS: app_id or token not configured")
            return ""

        voice = voice or self.default_voice

        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(f"TTS text truncated from {len(text)} to {self.MAX_TEXT_LENGTH} chars")
            text = text[: self.MAX_TEXT_LENGTH]

        # Check cache
        key = self._cache_key(text, voice)
        ext = f".{self.encoding}"
        cache_file = self.cache_dir / f"{key}{ext}"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            logger.debug(f"TTS cache hit: {cache_file}")
            return str(cache_file)

        try:
            payload = {
                "app": {
                    "appid": self.app_id,
                    "token": "access_token",
                    "cluster": self.cluster,
                },
                "user": {
                    "uid": "nanobot_zero",
                },
                "audio": {
                    "voice_type": voice,
                    "encoding": self.encoding,
                    "speed_ratio": speed_ratio,
                },
                "request": {
                    "reqid": str(uuid.uuid4()),
                    "text": text,
                    "operation": "query",
                },
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer;{self.token}",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )

                data = response.json()
                code = data.get("code", -1)

                if code != 3000:
                    msg = data.get("message", "unknown error")
                    logger.error(f"Volcengine TTS error (code={code}): {msg}")
                    return ""

                audio_b64 = data.get("data", "")
                if not audio_b64:
                    logger.error("Volcengine TTS returned empty audio data")
                    return ""

                audio_bytes = base64.b64decode(audio_b64)
                cache_file.write_bytes(audio_bytes)
                logger.info(f"TTS synthesized ({len(audio_bytes)} bytes, voice={voice}): {cache_file}")
                return str(cache_file)

        except Exception as e:
            logger.error(f"Volcengine TTS error: {e}")
            return ""
