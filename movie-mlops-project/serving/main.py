from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime

# 로깅 설정 - 서버에서 일어나는 모든 일을 기록
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨 이상의 로그만 출력
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 로그 출력 형식
)
logger = logging.getLogger(__name__)  # 현재 모듈용 로거 생성

# FastAPI 애플리케이션 인스턴스 생성
# Swagger문서의 맨 위에 표시 됨.
app = FastAPI(
    title="Movie Rating Prediction API", # API 제목
    description="TMDB 데이터를 활용한 영화 평점 예측 서비스",   # API 설명
    version="0.1.0",    # 초기 버전 - major(큰 변환).minor(기능 추가).patch버전(버그 수정)
    docs_url="/docs",   # Swagger UI 경로
    redoc_url="/redoc",  # ReDoc UI 경로
)

# CORS 설정 - 웹 브라우저에서 API 호출을 허용하기 위한 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 접근 허용 (개발용)
    allow_credentials=True,  # 쿠키 포함 요청 허용
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# 엔드 포인트 설정: 클라이언트가 우리 서버에 요청을 보내는 주소
@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 기본 정보 제공
    서버가 정상 작동하는지 확인하는 기본 페이지
    """
    logger.info("루트 엔드포인트 접근")  # 접근 로그 기록
    return {
        "message": "Movie Rating Prediction API",
        "version:": "0.1.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),  # 현재 시간 추가
        "docs": "/docs",  # API 문서 링크 제공
    }


@app.get("/health")
async def health_check():
    """
    헬스체크 엔드포인트
    서버 상태를 확인하기 위한 간단한 엔드포인트
    로드밸런서나 모니터링 도구에서 주기적으로 호출
    """
    logger.info("헬스체크 요청 수신")  # 헬스체크 로그 기록
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
    }

# 개발용 서버 실행 코드
if __name__ == "__main__":
    logger.info("FastAPI 서버 시작")  # 서버 시작 로그
    uvicorn.run(
        "main:app",      # 앱 경로
        host="0.0.0.0",    # 모든 네트워크 인터페이스에서 접근 허용
        port=8000,
        reload=True,
        log_level="info",  # uvicorn 로그 레벨 설정
    )