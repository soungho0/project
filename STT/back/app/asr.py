import os, uuid, tempfile
from fastapi import APIRouter, UploadFile, File
from .schemas import TranscriptResponse
from ..models.whisper_loader import run_asr
from .utils import merge_asr_diar

# pyannote가 없거나 토큰 문제여도 전체가 죽지 않게 import 자체도 안전 처리
try:
    from ..models.pyannote_loader import diarize
except Exception as _e:
    diarize = None

router = APIRouter(prefix="/asr", tags=["asr"])

@router.post("/file", response_model=TranscriptResponse)
async def transcribe_file(file: UploadFile = File(...)):
    raw = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
        tmpf.write(raw); tmp_path = tmpf.name
    try:
        # 1) STT (항상 시도)
        asr_segments, lang = run_asr(tmp_path)

        # 2) 화자분리: 가능하면 시도, 실패해도 STT만 반환
        merged = asr_segments
        if diarize is not None:
            try:
                spk_segments = diarize(tmp_path)
                merged = merge_asr_diar(asr_segments, spk_segments)
            except Exception as e:
                print(f"[DIARIZE] skipped: {e}")

        duration = max([s["end"] for s in merged]) if merged else 0.0
        return {"segments": merged, "language": lang, "duration": duration}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
