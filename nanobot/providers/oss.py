"""Aliyun OSS media upload utility.

Uploads images from chat channels to OSS and returns presigned URLs.
Used to pass image URLs to multimodal LLMs instead of base64 encoding,
which saves tokens and works across conversation history turns.

Presigned URLs are used because the bucket may be private. The ``oss_key``
(object key) is stored persistently; a fresh presigned URL is generated
on every LLM call so it never expires mid-conversation.
"""

import mimetypes
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

# Lazy import — oss2 is optional; only needed when OSS is configured.
_oss2 = None


def _get_oss2():
    global _oss2
    if _oss2 is None:
        try:
            import oss2
            _oss2 = oss2
        except ImportError:
            raise ImportError(
                "oss2 package is required for OSS uploads. "
                "Install it with: pip install oss2"
            )
    return _oss2


class OSSUploader:
    """Upload files to Aliyun OSS and return presigned URLs.

    Files are stored under ``{prefix}/{YYYY/MM/DD}/{uuid}.{ext}``.
    Presigned URLs are generated via ``sign_url`` so private buckets work.
    """

    # Default presign validity: 1 hour (enough for a single LLM call).
    DEFAULT_EXPIRES = 3600

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        access_key_secret: str,
        prefix: str = "nanobot/media",
    ):
        self.endpoint = endpoint
        self.bucket_name = bucket
        self.prefix = prefix

        oss2 = _get_oss2()
        auth = oss2.Auth(access_key_id, access_key_secret)
        self._bucket = oss2.Bucket(auth, endpoint, bucket)

    def sign_url(self, key: str, expires: int | None = None) -> str:
        """Generate a presigned GET URL for an existing object.

        Args:
            key: The OSS object key (e.g. ``nanoteacher/2026/02/23/abc.png``).
            expires: Validity in seconds (default :attr:`DEFAULT_EXPIRES`).

        Returns:
            Presigned URL string.
        """
        if expires is None:
            expires = self.DEFAULT_EXPIRES
        return self._bucket.sign_url("GET", key, expires)

    def upload(self, local_path: str | Path) -> dict | None:
        """Upload a local file to OSS.

        Args:
            local_path: Path to the local file.

        Returns:
            ``{"key": oss_key, "url": presigned_url}`` on success, None on failure.
        """
        p = Path(local_path)
        if not p.is_file():
            logger.warning(f"OSS upload skipped: file not found: {p}")
            return None

        ext = p.suffix or ".bin"
        date_path = datetime.now().strftime("%Y/%m/%d")
        key = f"{self.prefix}/{date_path}/{uuid.uuid4().hex[:12]}{ext}"

        mime, _ = mimetypes.guess_type(str(p))

        try:
            headers = {}
            if mime:
                headers["Content-Type"] = mime

            self._bucket.put_object_from_file(key, str(p), headers=headers)
            signed_url = self.sign_url(key)
            logger.info(f"OSS upload OK: {p.name} → key={key}")
            return {"key": key, "url": signed_url}
        except Exception as e:
            logger.error(f"OSS upload failed for {p}: {e}")
            return None


# Module-level singleton, created lazily on first use.
_uploader: OSSUploader | None = None
_init_attempted: bool = False


def get_uploader() -> OSSUploader | None:
    """Get the global OSS uploader (lazy init from config).

    Returns None if OSS is not configured.
    """
    global _uploader, _init_attempted
    if _init_attempted:
        return _uploader
    _init_attempted = True

    try:
        from nanobot.config.loader import load_config

        config = load_config()
        oss_cfg = config.tools.oss

        if not oss_cfg.access_key_id or not oss_cfg.bucket:
            logger.debug("OSS not configured — media will use base64 fallback")
            return None

        _uploader = OSSUploader(
            endpoint=oss_cfg.endpoint,
            bucket=oss_cfg.bucket,
            access_key_id=oss_cfg.access_key_id,
            access_key_secret=oss_cfg.access_key_secret,
            prefix=oss_cfg.prefix,
        )
        logger.info(f"OSS uploader ready: {oss_cfg.bucket}.{oss_cfg.endpoint}/{oss_cfg.prefix}/")
        return _uploader
    except Exception as e:
        logger.warning(f"OSS uploader init failed (falling back to base64): {e}")
        return None


async def upload_to_oss(local_path: str | Path) -> str | None:
    """Upload a file to OSS if configured, otherwise return None.

    This is the main entry point for channels.  Returns only the presigned
    URL string for backward compatibility.  Usage::

        url = await upload_to_oss("/path/to/image.jpg")
        media_ref = url or str(local_path)  # fallback to local path
    """
    uploader = get_uploader()
    if uploader is None:
        return None
    # oss2 is synchronous; run in thread to avoid blocking the event loop.
    import asyncio
    result = await asyncio.to_thread(uploader.upload, local_path)
    return result["url"] if result else None
