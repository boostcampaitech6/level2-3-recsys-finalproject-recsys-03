from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from api import router


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

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="some-secret-key")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
