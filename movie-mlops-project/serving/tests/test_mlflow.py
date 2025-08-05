import sys
import os
import logging

# 상위 디렉토리의 services 모듈을 import 할 수 있도록 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.mlflow_service import MLflowModelService

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_mlflow_connection():
    """
    MLflow 서버 연결 테스트
    """
    print("🧪 MLflow 연결 테스트")
    print("=" * 40)
    
    try:
        # 서비스 객체 생성
        print("📦 MLflowModelService 생성 중...")
        service = MLflowModelService()
        
        # 연결 상태 확인
        print("\n🔍 MLflow 서버 연결 확인 중...")
        result = service.check_mlflow_connection()
        
        print(f"\n📊 연결 테스트 결과:")
        print(f"   성공 여부: {result['success']}")
        print(f"   메시지: {result['message']}")
        print(f"   Tracking URI: {result['tracking_uri']}")
        print(f"   실험 수: {result['experiments_count']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def test_production_model_check():
    """
    Production 모델 존재 여부 테스트
    """
    print("\n🧪 Production 모델 존재 확인 테스트")
    print("=" * 40)
    
    try:
        service = MLflowModelService()
        
        print("🔍 Production 모델 존재 확인 중...")
        result = service.check_production_model_exists()
        
        print(f"\n📊 Production 모델 확인 결과:")
        print(f"   존재 여부: {result['exists']}")
        print(f"   메시지: {result['message']}")
        
        if result['exists']:
            print(f"   모델 버전: {result['version']}")
            print(f"   모델 URI: {result['model_uri']}")
        
        return result['exists']
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def test_model_loading():
    """
    모델 로딩 테스트 (메인 테스트)
    """
    print("\n🧪 Production 모델 로딩 테스트")
    print("=" * 40)
    
    try:
        service = MLflowModelService()
        
        print("🤖 Production 모델 로딩 중...")
        result = service.load_production_model()
        
        print(f"\n📊 모델 로딩 결과:")
        print(f"   성공 여부: {result['success']}")
        print(f"   메시지: {result['message']}")
        print(f"   모델 로드됨: {result['model_loaded']}")
        print(f"   모델 버전: {result['model_version']}")
        
        # 모델 정보 상세 조회
        if result['success']:
            print(f"\n📋 모델 상세 정보:")
            model_info = service.get_model_info()
            
            for key, value in model_info.items():
                print(f"   {key}: {value}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def run_all_tests():
    """
    전체 테스트 실행
    """
    print("🚀 MLflow 모델 서비스 전체 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("MLflow 연결", test_mlflow_connection),
        ("Production 모델 확인", test_production_model_check),
        ("모델 로딩", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        results.append((test_name, result))
        
        if result:
            print(f"✅ {test_name} 테스트 성공!")
        else:
            print(f"❌ {test_name} 테스트 실패!")
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    print("📊 전체 테스트 결과 요약")
    print("=" * 60)
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 모든 테스트 성공! MLflow 모델 서비스가 정상 작동합니다.")
        print("\n🎯 다음 단계:")
        print("   → movie_service.py에서 MLflowModelService 사용")
        print("   → 예측 기능 구현")
        return True
    else:
        print(f"\n⚠️ {total_count - success_count}개 테스트 실패")
        print("\n💡 해결 방법:")
        print("   1. MLflow 서버가 실행 중인지 확인")
        print("   2. register_mlflow.py로 Production 모델 등록 확인")
        print("   3. 환경변수 MLFLOW_TRACKING_URI 확인")
        return False

if __name__ == "__main__":
    # 환경변수 가이드
    print("💡 MLflow 테스트 실행 전 확인사항:")
    print("=" * 50)
    print("1. MLflow 서버 실행 여부:")
    print("   mlflow server --host 0.0.0.0 --port 5001")
    print("2. 환경변수 설정:")
    print("   export MLFLOW_TRACKING_URI=http://localhost:5001")
    print("3. Production 모델 등록:")
    print("   register_mlflow.py 스크립트 실행")
    print("")
    
    # 사용자 입력 대기
    input("준비가 완료되면 Enter를 눌러 테스트를 시작하세요...")
    
    # 전체 테스트 실행
    run_all_tests()