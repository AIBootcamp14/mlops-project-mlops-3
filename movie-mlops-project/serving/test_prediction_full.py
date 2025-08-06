import logging
import sys
import os

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from services.prediction_service import SimplePredictionService

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def test_full_prediction_workflow():
    """전체 예측 워크플로우 테스트"""
    print("🚀 전체 예측 워크플로우 테스트")
    print("=" * 50)
    
    # 서비스 생성 및 초기화
    print("1️⃣ 서비스 초기화...")
    service = SimplePredictionService()
    
    init_result = service.initialize()
    if not init_result['success']:
        print(f"❌ 초기화 실패: {init_result['message']}")
        return False
    
    print(f"✅ 초기화 성공!")
    print(f"   모델 버전: {init_result.get('model_version')}")
    print(f"   영화 데이터: {init_result.get('data_count')}개")
    
    # 전체 예측 실행
    print(f"\n2️⃣ 전체 영화 예측 실행...")
    pred_result = service.predict_all()
    
    if not pred_result['success']:
        print(f"❌ 예측 실패: {pred_result['message']}")
        return False
    
    print(f"✅ 예측 성공!")
    print(f"   예측 완료: {pred_result['sample_count']}개 영화")
    
    # 예측 결과 조회
    print(f"\n3️⃣ 예측 결과 조회...")
    predictions = service.get_predictions()
    
    if not predictions['available']:
        print(f"❌ 예측 결과 조회 실패: {predictions['message']}")
        return False
    
    print(f"✅ 예측 결과 조회 성공!")
    print(f"   총 예측 결과: {predictions['sample_count']}개")
    
    # 상위 10개 영화 조회
    print(f"\n4️⃣ 예측 평점 상위 10개 영화...")
    top_movies = service.get_top_movies(10)
    
    if not top_movies['available']:
        print(f"❌ 상위 영화 조회 실패: {top_movies['message']}")
        return False
    
    print(f"✅ 상위 영화 조회 성공!")
    print(f"\n🏆 예측 평점 TOP 10:")
    for movie in top_movies['top_movies']:
        print(f"   {movie['rank']:2d}. 영화 ID {movie['movie_id']:6d}: {movie['predicted_rating']:.2f}")
    
    # 예측 통계
    print(f"\n5️⃣ 예측 통계...")
    ratings = [result['predicted_rating'] for result in predictions['results']]
    
    import statistics
    avg_rating = statistics.mean(ratings)
    max_rating = max(ratings)
    min_rating = min(ratings)
    
    print(f"   평균 예측 평점: {avg_rating:.2f}")
    print(f"   최고 예측 평점: {max_rating:.2f}")
    print(f"   최저 예측 평점: {min_rating:.2f}")
    
    # 서비스 최종 상태
    print(f"\n6️⃣ 서비스 최종 상태...")
    status = service.get_status()
    
    print(f"   서비스 준비: {'✅' if status['service_ready'] else '❌'}")
    print(f"   모델 로드: {'✅' if status['model_loaded'] else '❌'}")
    print(f"   데이터 로드: {'✅' if status['data_loaded'] else '❌'}")
    print(f"   예측 완료: {'✅' if status['predictions_available'] else '❌'}")
    print(f"   예측 개수: {status['sample_count']}개")
    
    print(f"\n🎉 전체 워크플로우 테스트 성공!")
    print(f"📋 완성된 기능:")
    print(f"   ✅ 모델 로딩")
    print(f"   ✅ 데이터 로딩") 
    print(f"   ✅ 예측 실행")
    print(f"   ✅ 결과 조회")
    print(f"   ✅ 상위 영화 추출")
    print(f"   ✅ 상태 모니터링")
    print(f"\n🎯 다음 단계: FastAPI 연동으로 웹 API 구축")
    
    return True

if __name__ == "__main__":
    test_full_prediction_workflow()