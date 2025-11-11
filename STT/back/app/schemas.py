# API에서 입출력으로 쓰는 데이터 구조(pydantic 모델)

from pydantic import BaseModel
from typing import List, Optional

class TranscriptSegment(BaseModel):
    start: float                # 구간 시작(초)
    end: float                  # 구간 끝(초)
    text: str                   # 전사 텍스트
    speaker: Optional[str] = None  # 화자 라벨(SPK1 등)

class TranscriptResponse(BaseModel):
    segments: List[TranscriptSegment]
    language: str               # 인식된 언어
    duration: float             # 전체 길이(초)
