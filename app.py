import io
import os

import soundfile as sf
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from huggingface_hub import login
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, VitsModel

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "facebook/mms-tts-fon")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))

if HF_TOKEN:
    login(token=HF_TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[startup] Chargement du modele {MODEL_ID} sur {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = VitsModel.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
model.eval()

SAMPLE_RATE = model.config.sampling_rate
NUM_SPEAKERS = getattr(model.config, "num_speakers", 1) or 1

app = FastAPI(title="TTS Fongbe", version="1.0.0")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    speaker_id: int | None = Field(default=None, ge=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "sample_rate": SAMPLE_RATE,
        "num_speakers": NUM_SPEAKERS,
    }


@app.post("/synthesize")
def synthesize(payload: TTSRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le texte est vide")

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Texte trop long: max {MAX_TEXT_LENGTH} caracteres",
        )

    if payload.speaker_id is not None and NUM_SPEAKERS <= 1:
        raise HTTPException(
            status_code=400,
            detail="Ce modele ne supporte pas plusieurs locuteurs",
        )

    if payload.speaker_id is not None and payload.speaker_id >= NUM_SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"speaker_id invalide (0 a {NUM_SPEAKERS - 1})",
        )

    try:
        inputs = tokenizer(text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if payload.speaker_id is not None:
            inputs["speaker_id"] = torch.tensor([payload.speaker_id], device=device)

        with torch.no_grad():
            waveform = model(**inputs).waveform

        audio = waveform.squeeze().cpu().numpy()
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=SAMPLE_RATE, format="WAV")

        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="tts.wav"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de synthese: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8007, reload=False)
