# 📁 diarize.py
# pyannote.audio를 이용해 화자 구간(시작/끝/화자명)만 추출하는 라우터

from fastapi import APIRouter, UploadFile, File
from back.models.pyannote_loader import diarize

router = APIRouter(prefix="/diarize", tags=["diarization"])

@router.post("/file")
async def diarize_file(file: UploadFile = File(...)):
    """
    업로드된 오디오 파일에서 화자 구간(turn)을 추출합니다.
    pyannote.audio 파이프라인을 사용합니다.
    """
    # 업로드된 오디오를 임시 파일로 저장
    import uuid
    tmp = f"/tmp/{uuid.uuid4().hex}.wav"
    with open(tmp, "wb") as f:
        f.write(await file.read())

    # 화자 분리 수행
    segments = diarize(tmp)

    # 응답 예시: [{"start":0.0,"end":3.1,"speaker":"SPEAKER_00"}, ...]
    return {"segments": segments, "count": len(segments)}
