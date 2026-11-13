"""FastAPI backend.

  MODEL_PATH=results/model.joblib uvicorn v2.serve:app --port 8000

Environment:
  MODEL_PATH       joblib saved by `v2.evaluate --save_model`
  EMBEDDER         "clip" (default) or "dummy" (tests only)
  ABSTAIN_BELOW    confidence below which the verdict is "uncertain" (default 0.65)
  ALLOWED_ORIGINS  comma-separated CORS origins (default "*")
  LOW_MEM          "1" loads CLIP in bfloat16 for small servers

Responds 503 if no trained model is found; there is no untrained fallback.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from v2.faces import FaceCropper, sample_face_crops
from v2.headmodel import pool_frames

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "results/model.joblib"))
EMBEDDER_KIND = os.environ.get("EMBEDDER", "clip")
ABSTAIN_BELOW = float(os.environ.get("ABSTAIN_BELOW", "0.65"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))

app = FastAPI(title="Compression-robust deepfake detection API")
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

_state: dict = {}


def _load():
    if "bundle" in _state:
        return _state
    if not MODEL_PATH.exists():
        raise HTTPException(
            503,
            f"No trained model found at {MODEL_PATH}. Train with v2/evaluate.py --save_model "
            f"and set MODEL_PATH. This server does not answer with untrained weights.",
        )
    from v2.embed import get_embedder

    _state["bundle"] = joblib.load(MODEL_PATH)
    import sklearn

    trained_with = _state["bundle"].get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        print(f"[serve] WARNING: model trained with scikit-learn {trained_with}, "
              f"serving with {sklearn.__version__}. Pin the same version in requirements_serve.txt.")
    _state["embedder"] = get_embedder(EMBEDDER_KIND)
    _state["cropper"] = FaceCropper()
    return _state


def frame_influence(head, frame_embs: np.ndarray) -> list[float]:
    """Leave-one-frame-out change in the ensemble logit. Positive pushes toward fake.
    Indicates which frames mattered, not where an artifact is within a frame."""
    n = len(frame_embs)
    full, _ = head.raw_logits(pool_frames(frame_embs[None]))
    out = []
    for t in range(n):
        rest = np.delete(frame_embs, t, axis=0)
        without, _ = head.raw_logits(pool_frames(rest[None]))
        out.append(float(full[0] - without[0]))
    return out


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "embedder": EMBEDDER_KIND,
        "abstain_below": ABSTAIN_BELOW,
    }


@app.post("/predict")
async def predict(file: UploadFile | None = File(default=None), url: str | None = Form(default=None)):
    if file is None and not url:
        raise HTTPException(400, "Provide a video file or a YouTube url.")
    state = _load()
    head = state["bundle"]["head"]
    work = Path(tempfile.mkdtemp(prefix="sig_"))
    try:
        if url:
            from ingestion.source_resolver import DownloadError, UnsupportedSourceError, resolve_source

            try:
                video_path = resolve_source(url).local_path
            except UnsupportedSourceError as e:
                raise HTTPException(400, str(e))
            except DownloadError as e:
                raise HTTPException(502, f"Could not download the video: {e}")
        else:
            video_path = work / (Path(file.filename or "upload.mp4").name)
            size = 0
            with open(video_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1024 * 1024:
                        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
                    f.write(chunk)

        crops, found = sample_face_crops(video_path, state["cropper"], n_frames=state["bundle"]["n_frames"])
        if crops is None:
            raise HTTPException(
                422, "Could not find a clear face in enough frames. Try a clip where a face is visible."
            )

        frame_embs = state["embedder"](crops)
        pred = head.predict(pool_frames(frame_embs[None]))
        p_fake = float(pred["p_fake"][0])
        confidence = max(p_fake, 1 - p_fake)
        abstained = confidence < ABSTAIN_BELOW
        verdict = "uncertain" if abstained else ("likely_fake" if p_fake >= 0.5 else "likely_real")

        notes = ["Informational only. Not a forensic or legal determination."]
        if found < state["bundle"]["n_frames"]:
            notes.append(f"A face was found in only {found} of the sampled frames.")

        return {
            "verdict": verdict,
            "fake_probability": round(p_fake, 4),
            "real_probability": round(1 - p_fake, 4),
            "confidence": round(confidence, 4),
            "member_disagreement": round(float(pred["disagreement"][0]), 4),
            "abstained": abstained,
            "frame_influence": [round(v, 4) for v in frame_influence(head, frame_embs)],
            "faces_found": int(found),
            "notes": notes,
            "model_info": {
                "backbone": state["bundle"]["embedder"],
                "detector": state["cropper"].name,
                "training_note": state["bundle"]["note"],
            },
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
