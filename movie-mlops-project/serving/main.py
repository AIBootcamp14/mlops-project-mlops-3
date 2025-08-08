from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys
from datetime import datetime
import os

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 예측 서비스 import
from services.prediction_service import SimplePredictionService

# ============================================================================
# 🔧 통합 로깅 설정 (uvicorn과 통합)
# ============================================================================
def setup_logging():
    """통합 로깅 설정 - uvicorn과 애플리케이션 로그를 모두 콘솔에 표시"""
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 로그 포맷 설정 (시간, 모듈명, 레벨, 메시지)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 루트 로거에 핸들러 추가
    root_logger.addHandler(console_handler)
    
    # 각 모듈별 로거 설정
    loggers = [
        'main',
        'services.prediction_service', 
        'services.mlflow_service',
        'services.data_service',
        'uvicorn.access',
        'uvicorn.error'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = True  # 부모 로거로 전파
    
    print("🔧 통합 로깅 설정 완료!")

# 애플리케이션 시작 전 로깅 설정
setup_logging()

# 현재 모듈용 로거 생성
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="🎬 Movie Rating Prediction API", 
    description="TMDB 데이터를 활용한 영화 평점 예측 서비스 (MLflow + XGBoost)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 예측 서비스 객체
prediction_service = None

@app.on_event("startup")
async def startup_event():
    """서버 시작시 예측 서비스 초기화 및 예측 실행"""
    global prediction_service
    
    logger.info("=" * 80)
    logger.info("🚀 Movie Rating Prediction FastAPI 서버 시작!")
    logger.info("=" * 80)
    
    try:
        # 1단계: 예측 서비스 생성 및 초기화
        logger.info("📦 1단계: SimplePredictionService 객체 생성 중...")
        prediction_service = SimplePredictionService()
        logger.info("✅ 예측 서비스 객체 생성 완료")
        
        logger.info("🤖 2단계: MLflow Production 모델 & CSV 데이터 로딩 중...")
        init_result = prediction_service.initialize()
        
        if not init_result['success']:
            logger.error(f"❌ 예측 서비스 초기화 실패: {init_result['message']}")
            logger.error("💡 해결 방법:")
            logger.error("   1. MLflow 서버가 실행 중인지 확인: http://localhost:5001")
            logger.error("   2. Production 모델이 등록되어 있는지 확인")
            logger.error("   3. tmdb_test.csv 파일 경로 확인")
            return
        
        logger.info("✅ 예측 서비스 초기화 완료!")
        logger.info(f"   🏷️ 모델 버전: {init_result.get('model_version')}")
        logger.info(f"   📊 로드된 영화 수: {init_result.get('data_count'):,}개")
        
        # 2단계: 전체 영화 예측 실행
        logger.info("🎯 3단계: 전체 영화 데이터 예측 실행 중...")
        logger.info("   (이 과정은 데이터 크기에 따라 몇 초~몇 분 소요됩니다)")
        
        pred_result = prediction_service.predict_all()
        
        if pred_result['success']:
            logger.info("🎉 전체 예측 완료!")
            logger.info(f"   📈 예측 완료된 영화: {pred_result['sample_count']:,}개")
            
            # 상위 5개 영화 미리보기
            top_movies = prediction_service.get_top_movies(5)
            if top_movies['available']:
                logger.info("🏆 예측 평점 상위 5개 영화 미리보기:")
                for movie in top_movies['top_movies']:
                    logger.info(f"   {movie['rank']}위. 영화 ID {movie['movie_id']}: ⭐ {movie['predicted_rating']:.2f}")
            
        else:
            logger.error(f"❌ 예측 실행 실패: {pred_result['message']}")
            return
            
        # 최종 서비스 상태 확인
        status = prediction_service.get_status()
        logger.info("📊 4단계: 서비스 최종 상태 확인")
        logger.info(f"   서비스 준비: {'✅ 완료' if status['service_ready'] else '❌ 실패'}")
        logger.info(f"   모델 로드: {'✅ 완료' if status['model_loaded'] else '❌ 실패'}")
        logger.info(f"   데이터 로드: {'✅ 완료' if status['data_loaded'] else '❌ 실패'}")
        logger.info(f"   예측 완료: {'✅ 완료' if status['predictions_available'] else '❌ 실패'}")
        
        logger.info("=" * 80)
        logger.info("🎊 Movie Rating Prediction API 서버 준비 완료!")
        logger.info("🌐 API 문서: http://localhost:8000/docs")
        logger.info("📊 Streamlit 대시보드: streamlit run streamlit_app.py")
        logger.info("=" * 80)
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ 서버 시작 중 치명적인 오류 발생: {e}")
        logger.error("💡 디버깅 정보:")
        logger.error(f"   오류 타입: {type(e).__name__}")
        logger.error(f"   오류 메시지: {str(e)}")
        logger.error("=" * 80)
        import traceback
        logger.error("🔍 상세 오류 정보:")
        logger.error(traceback.format_exc())

# 기존 엔드포인트들...
@app.get("/")
async def root():
    """루트 엔드포인트 - API 기본 정보 제공"""
    logger.info("📝 루트 엔드포인트 접근")
    
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
    """헬스체크 엔드포인트"""
    logger.info("💊 헬스체크 요청 수신")
    
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

# 예측 관련 엔드포인트
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
        
        logger.info(f"📊 예측 결과 조회: {predictions['sample_count']:,}개")
        
        return {
            "success": True,
            "message": f"{predictions['sample_count']:,}개 영화 예측 결과",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_count": predictions['sample_count'],
                "predictions": predictions['results']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 예측 결과 조회 실패: {e}")
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
        
        logger.info(f"🏆 상위 {limit}개 영화 조회")
        
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
        logger.error(f"❌ 상위 영화 조회 실패: {e}")
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
        
        logger.info("📈 예측 통계 조회")
        
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
        logger.error(f"❌ 통계 조회 실패: {e}")
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
        logger.error(f"❌ 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")

# 개발용 서버 실행 코드
if __name__ == "__main__":
    logger.info("🎬 Movie Rating Prediction FastAPI 서버 시작")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 로깅 안정성을 위해 reload 비활성화
        log_level="info",
        access_log=True,
    )