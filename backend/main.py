from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from api import router
from backend.in_memory import model_memory

# print(dir(backend))
# members = inspect.getmembers(backend)
# for member in members:
#     print(member)

# Dev, Prod 구분에 따라 어떻게 구현할 것인가?
# Data Input / Output 고려
# Database ==> Cloud Database(AWS Aurora, GCP Cloud SQL)
# API 서버 모니터링
# API 부하 테스트
# Test Code

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # 데이터베이스 테이블 생성
#     logger.info("Creating database table")
#     SQLModel.metadata.create_all(engine)
#     yield

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://accounts.spotify.com",
    "https://accounts.spotify.com/api/token",
    "https://accounts.spotify.com/authorize",
    "https://api.spotify.com/v1/"
]

tag_model_dict = None
cbf_model_dict = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Start Up Event")
    model_memory.load_tag_model_memory()
    model_memory.load_cbf_model_memory()

    # 이전은 앱 시작 전 동작
    yield
    # 이후는 앱 종료 때 동작
    
    print("Shutdown Event!")
    # PyTorch 모델은 직접적인 메모리 해제 함수가 없음, 모델과 데이터에 None을 할당하여 참조 제거
    tag_model_dict = None
    cbf_model_dict = None

app = FastAPI(lifespan = lifespan)

app.add_middleware(SessionMiddleware, secret_key="some-secret-key")
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
