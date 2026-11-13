FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements_serve.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements_serve.txt

COPY --chown=user v2 ./v2
COPY --chown=user ingestion ./ingestion
COPY --chown=user results/model.joblib ./results/model.joblib

USER user
RUN python -c "from v2.faces import FaceCropper; print('detector:', FaceCropper().name)" \
    && python -c "from v2.embed import ClipEmbedder; ClipEmbedder(); print('clip ok')"

ENV MODEL_PATH=results/model.joblib EMBEDDER=clip
EXPOSE 7860
CMD ["uvicorn", "v2.serve:app", "--host", "0.0.0.0", "--port", "7860"]
