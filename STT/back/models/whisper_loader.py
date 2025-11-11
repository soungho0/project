import os

# 🔧 OpenMP 중복 로드 경고 방지 (Windows 환경용)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pyannote.audio import Pipeline


# ==============================
# 환경 변수 기반 설정
# ==============================
ASR_IMPL = os.getenv("ASR_IMPL", "faster-whisper").lower()
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "base")  # tiny, base, small, medium, large-v3 등 가능
LANG_HINT = os.getenv("LANG_HINT", "auto")
HF_TOKEN = os.getenv("HF_TOKEN")  # 선택적 토큰 (없으면 public model 사용)

_whisper_model = None
_diarization_model = None


# ==============================
# ① Whisper (STT) 모델 로드
# ==============================
def load_asr():
    """Whisper STT 모델 로드 (faster-whisper 권장)"""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    if ASR_IMPL == "faster-whisper":
        from faster_whisper import WhisperModel

        # ✅ 최신 구조: "Systran/..." 제거
        model_name = ASR_MODEL_SIZE
        print(f"[INFO] Loading faster-whisper model: {model_name}")

        # GPU 감지 — nvidia-smi 없으면 CPU로 실행
        try:
            has_gpu = os.system("nvidia-smi >nul 2>&1") == 0
        except Exception:
            has_gpu = False

        device = "cuda" if has_gpu else "cpu"

        # ⚙️ cudnn DLL 오류 방지를 위해 CPU로 강제 설정 가능
        if device == "cuda":
            print("[INFO] CUDA detected, but cudnn DLL missing may trigger warning.")
        else:
            print("[INFO] Running in CPU mode (no GPU detected).")

        _whisper_model = WhisperModel(model_name, device="cpu", compute_type="float32")
    else:
        import whisper
        print(f"[INFO] Loading OpenAI Whisper model: {ASR_MODEL_SIZE}")
        _whisper_model = whisper.load_model(ASR_MODEL_SIZE)

    print("[INFO] Whisper model loaded successfully ✅")
    return _whisper_model


# ==============================
# ② Pyannote (화자분리) 로더
# ==============================
def load_diarizer():
    """화자 분리 모델 로드 (토큰 없는 환경에서도 안전하게 작동)"""
    global _diarization_model
    if _diarization_model is not None:
        return _diarization_model

    try:
        if HF_TOKEN:
            print("[INFO] Using HF_TOKEN for diarization model.")
            _diarization_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            )
        else:
            print("[INFO] Using public diarization model (no token).")
            _diarization_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )

        if _diarization_model is None:
            raise RuntimeError("Diarization model load returned None.")
        print("[INFO] Diarization model loaded successfully ✅")

    except Exception as e:
        print(f"[ERROR] Failed to load diarization model: {e}")
        _diarization_model = None

    return _diarization_model


# ==============================
# ③ 화자 분리 실행 함수
# ==============================
def run_diarization(audio_path: str):
    """오디오 파일에서 화자 구간 추출"""
    diarizer = load_diarizer()
    if diarizer is None:
        print("[WARN] Diarizer not loaded. Skipping speaker separation.")
        return []

    try:
        diarization = diarizer(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            })
        print(f"[INFO] Diarization complete ({len(segments)} segments).")
        return segments
    except Exception as e:
        print(f"[ERROR] Diarization failed: {e}")
        return []


# ==============================
# ④ STT + 화자분리 통합 실행
# ==============================
def run_asr(audio_path: str):
    """STT + (가능하면) 화자 분리까지 통합"""
    model = load_asr()

    # === 1️⃣ STT 수행 ===
    print("[INFO] Running STT...")
    if ASR_IMPL == "faster-whisper":
        segments, info = model.transcribe(
            audio_path,
            language=None if LANG_HINT == "auto" else LANG_HINT
        )
        stt_segments = [{"start": float(s.start), "end": float(s.end), "text": s.text} for s in segments]
        lang = info.language or LANG_HINT
    else:
        import whisper
        res = model.transcribe(
            audio_path,
            language=None if LANG_HINT == "auto" else LANG_HINT
        )
        stt_segments = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
                        for s in res["segments"]]
        lang = res.get("language", LANG_HINT)

    # === 2️⃣ 화자 분리 시도 ===
    diar_segments = run_diarization(audio_path)
    if not diar_segments:
        print("[INFO] No diarization segments. Returning plain STT output.")
        merged = [{"speaker": "화자미상", **s} for s in stt_segments]
        return merged, lang

    # === 3️⃣ 화자 구간 매칭 ===
    print("[INFO] Matching speakers to STT segments...")
    merged = []
    for stt in stt_segments:
        speaker = None
        for dia in diar_segments:
            if dia["start"] <= stt["start"] <= dia["end"]:
                speaker = dia["speaker"]
                break
        merged.append({
            "speaker": speaker or "화자미상",
            "start": stt["start"],
            "end": stt["end"],
            "text": stt["text"]
        })

    print(f"[INFO] STT+Speaker alignment complete ({len(merged)} segments).")
    return merged, lang
