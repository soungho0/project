# back/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ 상대 임포트가 가장 안전 (실행 경로에 덜 민감)
from .asr import router as asr_router

# diarize 라우터는 선택 (미구현/미설치여도 서버가 뜨게)
try:
    from .diarize import router as diarize_router
except Exception:
    diarize_router = None

app = FastAPI(title="STT + Diarization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 라우터 등록
app.include_router(asr_router)
if diarize_router:
    app.include_router(diarize_router)

@app.on_event("startup")
async def _print_routes():
    # 서버 시작 시 등록된 라우트 경로를 콘솔에 출력해서 404 디버깅에 도움
    print("=== Mounted routes ===")
    for r in app.routes:
        try:
            print(r.methods, r.path)
        except Exception:
            pass
    print("======================")

@app.get("/")
def root():
    return {"message": "STT + Diarization server is running 🚀"}
