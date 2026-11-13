"""Face detection and frame sampling.

Detector: OpenCV YuNet (FaceDetectorYN). The ONNX model is downloaded once from
opencv_zoo. If that fails, a Haar cascade is used as a fallback; check
FaceCropper.name to see which one is active.

Frames are read in one sequential pass (grab() skips, retrieve() decodes only the
wanted frames), which is much faster than seeking per frame.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"
_MIN_MODEL_BYTES = 100_000
_DETECT_MAX_SIDE = 960


def _ensure_yunet() -> Path | None:
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > _MIN_MODEL_BYTES:
        return YUNET_PATH
    try:
        urllib.request.urlretrieve(YUNET_URL, str(YUNET_PATH))
    except Exception as e:
        print(f"[faces] YuNet download failed ({e}); falling back to Haar.")
        return None
    if YUNET_PATH.stat().st_size <= _MIN_MODEL_BYTES:
        print("[faces] YuNet file is too small; falling back to Haar.")
        YUNET_PATH.unlink(missing_ok=True)
        return None
    return YUNET_PATH


class FaceCropper:
    def __init__(self, size: int = 224, pad: float = 0.3, score_threshold: float = 0.7):
        self.size, self.pad = size, pad
        self._yunet = None
        self._haar = None
        path = _ensure_yunet() if hasattr(cv2, "FaceDetectorYN") else None
        if path is not None:
            self._yunet = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), score_threshold, 0.3, 5000
            )
            self.name = "yunet"
        else:
            data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if not data_dir:
                raise RuntimeError(
                    "No face detector available. Install OpenCV 4: "
                    "pip install 'opencv-python-headless>=4.9,<5'"
                )
            self._haar = cv2.CascadeClassifier(
                str(Path(data_dir) / "haarcascade_frontalface_default.xml")
            )
            if self._haar.empty():
                raise RuntimeError("Could not load Haar cascade.")
            self.name = "haar"

    def _largest_box(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, _DETECT_MAX_SIDE / max(h, w))
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale) if scale < 1 else frame_bgr
        if self._yunet is not None:
            self._yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self._yunet.detect(small)
            if faces is None or len(faces) == 0:
                return None
            f = max(faces, key=lambda r: r[2] * r[3])
            x, y, bw, bh = f[0], f[1], f[2], f[3]
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            dets = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(dets) == 0:
                return None
            x, y, bw, bh = max(dets, key=lambda d: d[2] * d[3])
        return (x / scale, y / scale, bw / scale, bh / scale)

    def crop(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """RGB uint8 (size, size, 3) crop of the largest face, or None."""
        box = self._largest_box(frame_bgr)
        if box is None:
            return None
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * (1 + 2 * self.pad)
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def sample_face_crops(
    video_path: str | Path,
    cropper: FaceCropper,
    n_frames: int = 16,
    min_faces: int = 4,
    max_seconds: float = 60.0,
) -> tuple[np.ndarray | None, int]:
    """Return (crops (n_frames, S, S, 3) or None, faces_found).

    Considers 3 * n_frames evenly spaced candidate frames and keeps the first
    n_frames with a detectable face. If between min_faces and n_frames faces are
    found, the crops are cycled to fill the sequence. Only the first max_seconds
    of the video are analysed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = min(total, int(fps * max_seconds))

    n_candidates = min(total, n_frames * 3)
    wanted = sorted(set(np.linspace(0, total - 1, n_candidates).astype(int).tolist()))
    wanted_set = set(wanted)
    last = wanted[-1]

    crops: list[np.ndarray] = []
    idx = 0
    while idx <= last and len(crops) < n_frames:
        if not cap.grab():
            break
        if idx in wanted_set:
            ok, frame = cap.retrieve()
            if ok:
                c = cropper.crop(frame)
                if c is not None:
                    crops.append(c)
        idx += 1
    cap.release()

    found = len(crops)
    if found < min_faces:
        return None, found
    if found < n_frames:
        crops = [crops[i % found] for i in range(n_frames)]
    return np.stack(crops, axis=0), found
