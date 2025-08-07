from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime
import sys
import os

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 예측 서비스 import
from services.prediction_service import SimplePredictionService

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

# 전역 예측 서비스 객체
prediction_service = None


@app.on_event("startup")
async def startup_event():
    """서버 시작시 예측 서비스 초기화 및 예측 실행"""
    global prediction_service
    
    logger.info("🚀 FastAPI 서버 시작 - 예측 서비스 초기화 중...")
    
    try:
        # 1단계: 예측 서비스 생성 및 초기화
        prediction_service = SimplePredictionService()
        init_result = prediction_service.initialize()
        
        if not init_result['success']:
            logger.error(f"❌ 예측 서비스 초기화 실패: {init_result['message']}")
            return
        
        logger.info("✅ 예측 서비스 초기화 완료!")
        logger.info(f"   모델 버전: {init_result.get('model_version')}")
        logger.info(f"   데이터 개수: {init_result.get('data_count')}")
        
        # 2단계: 전체 영화 예측 실행
        logger.info("🤖 서버 시작시 전체 예측 실행 중...")
        pred_result = prediction_service.predict_all()
        
        if pred_result['success']:
            logger.info(f"✅ 예측 완료! {pred_result['sample_count']}개 영화")
        else:
            logger.error(f"❌ 예측 실행 실패: {pred_result['message']}")
            
    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        

# 엔드 포인트 설정: 클라이언트가 우리 서버에 요청을 보내는 주소
@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 기본 정보 제공
    서버가 정상 작동하는지 확인하는 기본 페이지
    """
    logger.info("루트 엔드포인트 접근")  # 접근 로그 기록
    
    # 서비스 상태 확인
    service_status = "ready" if prediction_service and prediction_service.is_ready else "initializing"
    predictions_count = 0
    if prediction_service and prediction_service.predictions:
        predictions_count = prediction_service.predictions['sample_count']
        
    return {
        "message": "🎬 Movie Rating Prediction API",
        "version": "1.0.0",
        "status": "running",
        "service_status": service_status,
        "predictions_available": predictions_count,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "predictions": "/predictions", 
            "top_movies": "/top-movies",
            "statistics": "/stats",
            "service_status": "/predict-status",
            "docs": "/docs"
        },
        "description": "MLflow + XGBoost 기반 영화 평점 예측 서비스"
    }


@app.get("/health")
async def health_check():
    """
    헬스체크 엔드포인트
    서버 상태를 확인하기 위한 간단한 엔드포인트
    로드밸런서나 모니터링 도구에서 주기적으로 호출
    """
    logger.info("헬스체크 요청 수신")  # 헬스체크 로그 기록
    
    if not prediction_service:
        return {
            "status": "starting",
            "timestamp": datetime.now().isoformat(),
            "message": "예측 서비스 초기화 중..."
        }
    
    service_status = prediction_service.get_status()
    
    overall_status = "healthy" if service_status['service_ready'] else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service_details": {
            "service_ready": service_status['service_ready'],
            "model_loaded": service_status['model_loaded'],
            "data_loaded": service_status['data_loaded'],
            "predictions_available": service_status['predictions_available'],
            "sample_count": service_status['sample_count']
        },
        "message": "영화 평점 예측 서비스 상태"
    }
    

# ========================================
# 예측 관련 엔드포인트
# ========================================

@app.get("/predictions")
async def get_all_predictions():
    """전체 예측 결과 조회"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="예측 서비스가 아직 준비되지 않았습니다.")
    
    if not prediction_service.is_ready:
        raise HTTPException(status_code=503, detail="예측 서비스가 초기화되지 않았습니다.")
    
    try:
        predictions = prediction_service.get_predictions()
        
        if not predictions['available']:
            raise HTTPException(status_code=404, detail="예측 결과를 찾을 수 없습니다.")
        
        logger.info(f"예측 결과 조회: {predictions['sample_count']}개")
        
        return {
            "success": True,
            "message": f"{predictions['sample_count']}개 영화 예측 결과",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_count": predictions['sample_count'],
                "predictions": predictions['results']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 결과 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예측 결과 조회 실패: {str(e)}")
    
    
@app.get("/top-movies")
async def get_top_movies(limit: int = Query(default=10, ge=1, le=50, description="상위 몇 개 영화를 가져올지")):
    """예측 평점 상위 영화 조회"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="예측 서비스가 아직 준비되지 않았습니다.")
    
    if not prediction_service.is_ready:
        raise HTTPException(status_code=503, detail="예측 서비스가 초기화되지 않았습니다.")
    
    try:
        top_movies = prediction_service.get_top_movies(limit)
        
        if not top_movies['available']:
            raise HTTPException(status_code=404, detail="예측 결과를 찾을 수 없습니다.")
        
        logger.info(f"상위 {limit}개 영화 조회")
        
        return {
            "success": True,
            "message": f"예측 평점 상위 {limit}개 영화",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "top_movies": top_movies['top_movies'],
                "total_movies": top_movies['total_count'],
                "showing_count": len(top_movies['top_movies'])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"상위 영화 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상위 영화 조회 실패: {str(e)}")
    

@app.get("/stats")
async def get_prediction_statistics():
    """예측 통계 정보 조회"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="예측 서비스가 아직 준비되지 않았습니다.")
    
    if not prediction_service.is_ready:
        raise HTTPException(status_code=503, detail="예측 서비스가 초기화되지 않았습니다.")
    
    try:
        predictions = prediction_service.get_predictions()
        
        if not predictions['available']:
            raise HTTPException(status_code=404, detail="예측 결과를 찾을 수 없습니다.")
        
        # 통계 계산
        ratings = [result['predicted_rating'] for result in predictions['results']]
        
        import statistics
        stats = {
            "total_movies": len(ratings),
            "average_rating": round(statistics.mean(ratings), 2),
            "max_rating": round(max(ratings), 2),
            "min_rating": round(min(ratings), 2),
            "median_rating": round(statistics.median(ratings), 2),
            "std_deviation": round(statistics.stdev(ratings), 2) if len(ratings) > 1 else 0
        }
        
        # 평점 구간별 분포
        rating_bins = {
            "8.0_이상": len([r for r in ratings if r >= 8.0]),
            "7.0_8.0": len([r for r in ratings if 7.0 <= r < 8.0]),
            "6.0_7.0": len([r for r in ratings if 6.0 <= r < 7.0]),
            "5.0_6.0": len([r for r in ratings if 5.0 <= r < 6.0]),
            "5.0_미만": len([r for r in ratings if r < 5.0])
        }
        
        logger.info("예측 통계 조회")
        
        return {
            "success": True,
            "message": "예측 통계 정보",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "statistics": stats,
                "distribution": rating_bins
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")
    
    
@app.get("/predict-status")
async def get_prediction_service_status():
    """예측 서비스 상태 조회"""
    if not prediction_service:
        return {
            "service_available": False,
            "message": "예측 서비스가 생성되지 않았습니다.",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        status = prediction_service.get_status()
        
        return {
            "service_available": True,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "message": "예측 서비스 상태 정보"
        }
        
    except Exception as e:
        logger.error(f"상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")


# 개발용 서버 실행 코드
if __name__ == "__main__":
    logger.info("🎬 Movie Rating Prediction FastAPI 서버 시작")  # 서버 시작 로그
    uvicorn.run(
        "main:app",      # 앱 경로
        host="0.0.0.0",    # 모든 네트워크 인터페이스에서 접근 허용
        port=8000,
        reload=True,
        log_level="info",  # uvicorn 로그 레벨 설정
    )