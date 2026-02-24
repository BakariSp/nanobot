"""Voice transcription provider using DashScope (百炼) Recognition."""

import asyncio
import os
from pathlib import Path

from loguru import logger


class DashScopeTranscriptionProvider:
    """
    Voice transcription using DashScope Recognition.call() — the official
    single-file approach from alibabacloud-bailian-speech-demo.

    Supports local audio files (ogg/opus/mp3/wav/etc).
    """

    # Map file extensions to DashScope format parameter
    FORMAT_MAP = {
        ".ogg": "opus",
        ".opus": "opus",
        ".mp3": "mp3",
        ".wav": "wav",
        ".m4a": "aac",
        ".aac": "aac",
        ".amr": "amr",
        ".pcm": "pcm",
        ".flac": "wav",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe a local audio file.

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self.api_key:
            logger.warning("STT skipped: DashScope API key is empty")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error(f"STT skipped: file not found {file_path}")
            return ""

        logger.info(f"STT starting: {path.name} (key={self.api_key[:8]}...)")
        return await asyncio.to_thread(self._transcribe_sync, path)

    def _transcribe_sync(self, path: Path) -> str:
        """Synchronous transcription using Recognition.call(file_path)."""
        try:
            import dashscope
            dashscope.api_key = self.api_key

            from dashscope.audio.asr import Recognition

            fmt = self.FORMAT_MAP.get(path.suffix.lower(), "opus")
            logger.debug(f"STT Recognition: model=paraformer-realtime-v2, format={fmt}, file={path}")

            recognition = Recognition(
                model="paraformer-realtime-v2",
                format=fmt,
                sample_rate=48000,
                language_hints=["zh", "en"],
                callback=None,
            )

            result = recognition.call(str(path))
            sentence_list = result.get_sentence()
            logger.debug(f"STT got {len(sentence_list) if sentence_list else 0} sentence(s)")

            if not sentence_list:
                logger.warning(f"STT returned no sentences for {path.name}")
                return ""

            text = "".join(s["text"] for s in sentence_list if "text" in s)
            if text:
                logger.info(f"STT OK: {text[:80]}...")
            return text

        except Exception as e:
            logger.error(f"STT error: {type(e).__name__}: {e}")
            return ""
