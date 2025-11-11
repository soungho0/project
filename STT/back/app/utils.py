# STT 세그먼트와 화자 세그먼트를 겹침(Overlap) 기반으로 매핑하는 도우미

from typing import List, Dict

def overlap(a, b) -> float:
    """두 구간 a=(s,e), b=(s,e)의 겹치는 길이(초) 계산"""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)

def merge_asr_diar(asr: List[Dict], spk: List[Dict]) -> List[Dict]:
    """ASR 세그먼트를 가장 많이 겹치는 화자 라벨과 매핑"""
    def label_for(t0, t1):
        best = (0.0, None)
        for s in spk:
            inter = overlap((t0, t1), (s["start"], s["end"]))
            if inter > best[0]:
                best = (inter, s["speaker"])
        return best[1] or "SPK?"
    return [{**seg, "speaker": label_for(seg["start"], seg["end"])} for seg in asr]
