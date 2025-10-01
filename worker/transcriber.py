import asyncio, base64, json, tempfile
from datetime import datetime
import redis.asyncio as redis
import whisper
from common.config import settings

AUDIO_QUEUE_KEY = "stt:audio"
TRANSCRIPT_CHANNEL = "stt:transcripts"

async def process_loop():
    r = redis.from_url(settings.REDIS_URL, decode_responses=False)
    model = whisper.load_model(settings.WHISPER_MODEL)
    while True:
        item = await r.blpop(AUDIO_QUEUE_KEY, timeout=5)
        if not item:
            await asyncio.sleep(0.05); continue
        _key, raw = item
        try:
            payload = json.loads(raw.decode("utf-8"))
            audio_bytes = base64.b64decode(payload["audio_b64"])
            language = payload.get("language")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(audio_bytes); tmp.flush()
                result = model.transcribe(tmp.name, language=language)
            text = (result.get("text") or "").strip()
            segments = result.get("segments", [])
            message = {
                "session_id": payload.get("session_id"),
                "text": text,
                "segments": [
                    {
                        "id": seg.get("id"),
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text"),
                        "avg_logprob": seg.get("avg_logprob"),
                        "no_speech_prob": seg.get("no_speech_prob"),
                    } for seg in segments
                ],
                "language": result.get("language"),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            await r.publish(TRANSCRIPT_CHANNEL, json.dumps(message).encode("utf-8"))
        except Exception as e:
            await r.publish(TRANSCRIPT_CHANNEL, json.dumps({"error": str(e)}).encode("utf-8"))

if __name__ == "__main__":
    asyncio.run(process_loop())
