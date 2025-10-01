import asyncio, base64, io, json, uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from common.config import settings

app = FastAPI(title="Whisper STT Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

redis_client: redis.Redis = redis.from_url(settings.REDIS_URL, decode_responses=False)
AUDIO_QUEUE_KEY = "stt:audio"
TRANSCRIPT_CHANNEL = "stt:transcripts"

@app.get("/health")
async def health():
    pong = await redis_client.ping()
    return {"status": "ok", "redis": pong}

@app.post("/transcribe")
async def transcribe_upload(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    sample_rate: Optional[int] = Form(default=16000),
    language: Optional[str] = Form(default=None),
):
    data = await file.read()
    payload = {
        "session_id": session_id or str(uuid.uuid4()),
        "content_type": file.content_type or "audio/wav",
        "sample_rate": sample_rate,
        "language": language,
        "audio_b64": base64.b64encode(data).decode("utf-8"),
    }
    await redis_client.rpush(AUDIO_QUEUE_KEY, json.dumps(payload).encode("utf-8"))
    return JSONResponse({"queued": True, "session_id": payload["session_id"]})

@app.websocket("/ws/audio")
async def ws_audio_in(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    buffer = io.BytesIO()
    sample_rate, language, content_type = 16000, None, "audio/wav"
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                buffer.write(message["bytes"])
            elif "text" in message and message["text"] is not None:
                text = message["text"].strip()
                if text.startswith("SET:"):
                    try:
                        k, v = text[4:].split("=", 1)
                        if k == "language": language = v or None
                        elif k == "sample_rate": sample_rate = int(v)
                        elif k == "content_type": content_type = v
                    except Exception:
                        pass
                    continue
                if text == "END":
                    audio_bytes = buffer.getvalue()
                    payload = {
                        "session_id": session_id,
                        "content_type": content_type,
                        "sample_rate": sample_rate,
                        "language": language,
                        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
                    }
                    await redis_client.rpush(AUDIO_QUEUE_KEY, json.dumps(payload).encode("utf-8"))
                    await websocket.send_text(json.dumps({"queued": True, "session_id": session_id}))
                    buffer.close(); buffer = io.BytesIO()
                else:
                    await websocket.send_text(json.dumps({"echo": text}))
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/transcripts")
async def ws_transcripts(websocket: WebSocket):
    await websocket.accept()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(TRANSCRIPT_CHANNEL)
    try:
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg.get("type") == "message":
                data = msg.get("data")
                if isinstance(data, (bytes, bytearray)):
                    await websocket.send_text(data.decode("utf-8"))
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        await pubsub.unsubscribe(TRANSCRIPT_CHANNEL)
        await pubsub.close()
