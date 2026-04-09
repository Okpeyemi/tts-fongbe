FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && pip install -r requirements.txt

COPY app.py .

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8007

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007", "--workers", "1"]
