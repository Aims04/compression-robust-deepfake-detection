"""Resolve a user-supplied input (local file or YouTube URL) to a local video file.

Other platforms are rejected with a clear message rather than scraped.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class UnsupportedSourceError(Exception):
    """The URL is from a platform that is not supported."""


class DownloadError(Exception):
    """yt-dlp failed to fetch a supported URL."""


YOUTUBE_PATTERNS = (
    re.compile(r"^https?://(www\.)?youtube\.com/watch\?v="),
    re.compile(r"^https?://youtu\.be/"),
    re.compile(r"^https?://(www\.)?youtube\.com/shorts/"),
)

KNOWN_UNSUPPORTED = {
    "instagram.com": "Instagram links are not supported. Download the video and upload it directly.",
    "tiktok.com": "TikTok links are not supported. Download the video and upload it directly.",
}


@dataclass
class ResolvedVideo:
    local_path: Path
    source_type: str  # "upload" or "youtube"
    original_reference: str


def resolve_source(user_input: str, upload_dir: Path | None = None) -> ResolvedVideo:
    if user_input.startswith(("http://", "https://")):
        return _resolve_url(user_input)
    return _resolve_upload(user_input, upload_dir)


def _resolve_upload(path_str: str, upload_dir: Path | None) -> ResolvedVideo:
    p = Path(path_str)
    if upload_dir is not None and not p.is_absolute():
        p = upload_dir / p
    if not p.exists():
        raise FileNotFoundError(f"Uploaded file not found: {p}")
    return ResolvedVideo(local_path=p, source_type="upload", original_reference=p.name)


def _resolve_url(url: str) -> ResolvedVideo:
    for domain, message in KNOWN_UNSUPPORTED.items():
        if domain in url:
            raise UnsupportedSourceError(message)
    if not any(pattern.match(url) for pattern in YOUTUBE_PATTERNS):
        raise UnsupportedSourceError("Only YouTube links and direct uploads are supported.")

    out_dir = Path(tempfile.mkdtemp(prefix="dfdetect_"))
    cmd = [
        "yt-dlp", "-f", "mp4/best", "--no-playlist", "--max-filesize", "500M",
        "-o", str(out_dir / "%(id)s.%(ext)s"), url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise DownloadError(f"yt-dlp failed to fetch {url}:\n{result.stderr}")

    downloaded = list(out_dir.glob("*.mp4"))
    if not downloaded:
        raise DownloadError(f"No mp4 was produced for {url}")
    return ResolvedVideo(local_path=downloaded[0], source_type="youtube", original_reference=url)
