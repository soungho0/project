import os
from typing import List, Dict
import torch

_HF_TOKEN = os.getenv("HF_TOKEN")
_diar_model = None

def load_diarizer():
    """HF 토큰으로 diarization 파이프라인 로드 (pyannote.audio 2.x/3.x 호환)"""
    global _diar_model
    if _diar_model is not None:
        return _diar_model

    if not _HF_TOKEN or len(_HF_TOKEN) <= 10:
        raise RuntimeError("HF_TOKEN missing; diarization disabled.")

    from pyannote.audio import Pipeline

    # 먼저 최신 문법(token=) 시도 → 실패시 구문(use_auth_token=) 재시도
    try:
        _diar_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=_HF_TOKEN
        )
    except TypeError:
        _diar_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=_HF_TOKEN
        )

    _diar_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return _diar_model

def diarize(audio_path: str) -> List[Dict]:
    """오디오에서 화자 구간 리스트[{start,end,speaker}] 반환"""
    pipe = load_diarizer()
    diar = pipe(audio_path)
    segments = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker)
        })
    return segments
