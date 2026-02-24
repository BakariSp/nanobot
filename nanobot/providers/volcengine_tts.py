"""Text-to-speech provider using Volcengine (火山引擎) 豆包语音合成 2.0.

HTTP V3 one-shot synthesis (unidirectional streaming collected into one file).
Docs: https://www.volcengine.com/docs/6561/1598757

Required config:
  tools.tts.volcengine_app_id     — APP ID from 火山引擎语音技术控制台
  tools.tts.volcengine_token      — Access Token from 控制台
  tools.tts.volcengine_resource_id — e.g. "seed-tts-2.0" (default)
  tools.tts.default_voice         — e.g. "zh_female_vv_uranus_bigtts"
"""

import base64
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


class VolcengineTTSProvider:
    """TTS via Volcengine Speech HTTP V3 API (豆包语音合成 2.0)."""

    API_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    MAX_TEXT_LENGTH = 3000

    def __init__(
        self,
        app_id: str,
        token: str,
        default_voice: str = "zh_female_vv_uranus_bigtts",
        resource_id: str = "seed-tts-2.0",
        encoding: str = "mp3",
        sample_rate: int = 24000,
    ):
        self.app_id = app_id
        self.token = token
        self.default_voice = default_voice
        self.resource_id = resource_id
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.cache_dir = Path.home() / ".nanobot" / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, text: str, voice: str, emotion: str | None = None) -> str:
        raw = f"{text}:{voice}:{emotion or ''}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return h

    async def _call_v3(
        self,
        text: str,
        voice: str,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        speech_rate: int | None = None,
    ) -> bytes:
        """Make a single V3 API call. Returns raw audio bytes (may be empty)."""
        headers = {
            "Content-Type": "application/json",
            "X-Api-App-Id": self.app_id,
            "X-Api-Access-Key": self.token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }

        audio_params: dict[str, Any] = {
            "format": self.encoding,
            "sample_rate": self.sample_rate,
        }
        if emotion:
            audio_params["emotion"] = emotion
        if emotion_scale is not None:
            audio_params["emotion_scale"] = emotion_scale
        if speech_rate is not None:
            audio_params["speech_rate"] = speech_rate

        payload = {
            "user": {"uid": "nanobot_zero"},
            "req_params": {
                "text": text,
                "speaker": voice,
                "audio_params": audio_params,
            },
        }

        audio_chunks: list[bytes] = []

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", self.API_URL, json=payload, headers=headers, timeout=60.0,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(f"Volcengine TTS V3 HTTP {response.status_code}: {body.decode(errors='replace')}")
                    return b""

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    code = chunk.get("code", -1)
                    if code == 0:
                        b64 = chunk.get("data", "")
                        if b64:
                            audio_chunks.append(base64.b64decode(b64))
                    elif code == 20000000:
                        usage = chunk.get("usage", {})
                        if usage:
                            logger.debug(f"TTS V3 usage: {usage}")
                    else:
                        msg = chunk.get("message", "unknown error")
                        logger.error(f"Volcengine TTS V3 error (code={code}): {msg}")
                        return b""

        return b"".join(audio_chunks)

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        speech_rate: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Synthesize text to speech via V3 API.

        If emotion is requested but the voice doesn't support it (returns empty
        audio), automatically retries without emotion as fallback.

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
        key = self._cache_key(text, voice, emotion)
        ext = f".{self.encoding}"
        cache_file = self.cache_dir / f"{key}{ext}"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            logger.debug(f"TTS cache hit: {cache_file}")
            return str(cache_file)

        try:
            audio_bytes = await self._call_v3(text, voice, emotion, emotion_scale, speech_rate)

            # Fallback: if emotion was requested but voice returned no audio, retry without
            if not audio_bytes and emotion:
                logger.warning(f"TTS voice={voice} may not support emotion={emotion}, retrying without emotion")
                audio_bytes = await self._call_v3(text, voice)
                # Use a cache key without emotion for the fallback result
                key = self._cache_key(text, voice, None)
                cache_file = self.cache_dir / f"{key}{ext}"

            if not audio_bytes:
                logger.error("Volcengine TTS V3 returned no audio data")
                return ""

            cache_file.write_bytes(audio_bytes)
            logger.info(f"TTS synthesized ({len(audio_bytes)} bytes, voice={voice}): {cache_file}")
            return str(cache_file)

        except Exception as e:
            logger.error(f"Volcengine TTS V3 error: {e}")
            return ""
