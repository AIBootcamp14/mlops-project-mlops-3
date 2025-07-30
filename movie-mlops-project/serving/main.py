from fastapi import FastAPI
import uvicorn

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="Movie Rating Prediction API",     # API 제목
    description="TMDB 데이터를 활용한 영화 평점 예측 서비스",        # API 설명
    version="0.1.0"     # 초기 버전
)

@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 기본 정보 제공
    서버가 정상 작동하는지 확인하는 기본 페이지
    """
    return {
        "message": "Movie Rating Prediction API",
        "version:": "0.1.0",
        "status": "running"
    }


@app.get
async def health_check():
    """
    헬스체크 엔드포인트
    서버 상태를 확인하기 위한 간단한 엔드포인트
    """
    return {"status": "healthy"}


# 개발용 서버 실행 코드
if __name__ == "__main__":
    uvicorn.run(
        "main:app",      # 앱 경로
        host="0.0.0.0",    # 모든 네트워크 인터페이스에서 접근 허용
        port=8000,
        reload=True
    )