# 병합된 세그먼트 -> SRT/VTT 텍스트로 변환(발표나 다운로드용)

from typing import List
import srt
from datetime import timedelta

def to_srt(segments: List[dict]) -> str:
    subs = []
    for i, seg in enumerate(segments, 1):
        subs.append(srt.Subtitle(
            index=i,
            start=timedelta(seconds=seg["start"]),
            end=timedelta(seconds=seg["end"]),
            content=f'{seg.get("speaker","SPK?")}: {seg["text"]}'
        ))
    return srt.compose(subs)

def to_vtt(segments: List[dict]) -> str:
    return "WEBVTT\n\n" + to_srt(segments).replace(",", ".")
