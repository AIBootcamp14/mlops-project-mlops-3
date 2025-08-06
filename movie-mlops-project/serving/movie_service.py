import logging
import os

# .env 파일 로드 (python-dotenv 필요: pip install python-dotenv)
try:
    from dotenv import load_dotenv
    
    # 여러 경로에서 .env 파일 찾기
    possible_paths = [
        ".env",           # 현재 디렉토리
        "../.env",        # 상위 디렉토리  
        "../../.env",     # 두 단계 상위 디렉토리
    ]
    
    env_loaded = False
    for path in possible_paths:
        if load_dotenv(dotenv_path=path):
            logging.info(f"📁 .env 파일 로드 완료: {path}")
            env_loaded = True
            break
    
    if not env_loaded:
        logging.warning("⚠️ .env 파일을 찾을 수 없음")
    
except ImportError:
    logging.warning("⚠️ python-dotenv가 설치되지 않음. 환경변수를 직접 설정하세요.")
    logging.warning("💡 설치 방법: pip install python-dotenv")
except Exception as e:
    logging.warning(f"⚠️ .env 파일 로드 실패: {e}")

# 로깅 설정 - 콘솔에 정보를 출력하도록 설정
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨 이상만 출력 (DEBUG는 출력 안함)
    format="%(asctime)s - %(levelname)s - %(message)s"  # 시간 - 레벨 - 메시지 형식
)
logger = logging.getLogger(__name__)  # 현재 모듈명으로 로거 생성

# MLflow 서비스 import
from services.mlflow_service import MLflowModelService

class MoviePredictionService:
    """
    영화 평점 예측 서비스 메인 클래스 (MLflow 기반)
    
    🔄 팀원 제안 반영:
    - S3 직접 접근 방식 → MLflow 모델 레지스트리 방식
    - 하드코딩된 모델 경로 → Production 스테이지 자동 추적
    - 수동 모델 관리 → 자동화된 모델 버전 관리
    
    ✅ 현재 구현 단계:
    - 1단계: MLflow 모델 서비스 통합 (완료)
    - 2단계: CSV 데이터 로드 기능 (예정)
    - 3단계: 예측 수행 기능 (예정)
    - 4단계: FastAPI 연동 (예정)
    """
    
    def __init__(self):
        """
        영화 평점 예측 서비스 초기화
        
        MLflow 기반 모델 서비스와 데이터 처리 준비
        """
        logger.info("🚀 MoviePredictionService 초기화 시작...")
        
        # === MLflow 모델 서비스 ===
        logger.info("🤖 MLflow 모델 서비스 초기화 중...")
        self.mlflow_service = MLflowModelService()
        
        # === 데이터 관련 설정 ===
        self.test_csv_path = "../preprocessing/result/tmdb_test.csv"
        self.test_data = None  # CSV 데이터가 저장될 공간 (2단계에서 사용)
        self.predictions = []  # 예측 결과들이 저장될 리스트 (3단계에서 사용)
        
        # === 서비스 상태 ===
        self.is_ready_for_prediction = False  # 예측 준비 완료 여부
        
        logger.info("✅ MoviePredictionService 초기화 완료")
        
    def check_service_health(self):
        """
        전체 서비스 상태 확인
        
        MLflow 연결, 모델 로드 등 모든 구성 요소의 상태를 확인합니다.
        FastAPI의 헬스체크 엔드포인트에서 사용할 예정입니다.
        
        Returns:
            dict: 전체 서비스 상태 정보
        """
        logger.info("🔍 서비스 상태 확인 중...")
        
        # MLflow 연결 상태 확인
        mlflow_connection = self.mlflow_service.check_mlflow_connection()
        
        # Production 모델 존재 여부 확인
        production_model = self.mlflow_service.check_production_model_exists()
        
        # 모델 로드 상태 확인
        model_info = self.mlflow_service.get_model_info()
        
        # 전체 상태 구성
        health_status = {
            'service_name': 'MoviePredictionService',
            'status': 'healthy' if (mlflow_connection['success'] and production_model['exists']) else 'unhealthy',
            'mlflow_connection': mlflow_connection,
            'production_model': production_model,
            'model_info': model_info,
            'ready_for_prediction': self.is_ready_for_prediction,
            'components': {
                'mlflow_service': '✅' if mlflow_connection['success'] else '❌',
                'production_model': '✅' if production_model['exists'] else '❌',
                'model_loaded': '✅' if model_info.get('model_loaded', False) else '❌',
                'data_service': '⏳' if not hasattr(self, 'data_service') else '✅',  # 2단계에서 구현 예정
            }
        }
        
        return health_status
        
    def initialize_model(self):
        """
        모델 초기화 (Production 모델 로드)
        
        MLflow 레지스트리에서 Production 스테이지의 모델을 로드합니다.
        
        Returns:
            dict: 모델 초기화 결과
        """
        logger.info("🤖 모델 초기화 시작...")
        
        # MLflow 서비스를 통해 Production 모델 로드
        result = self.mlflow_service.load_production_model()
        
        if result['success']:
            logger.info("✅ 모델 초기화 완료!")
            # 향후 예측 준비 상태 업데이트 (데이터 서비스도 준비되면)
            # self.is_ready_for_prediction = True  # 2단계에서 활성화 예정
        else:
            logger.error("❌ 모델 초기화 실패!")
            
        return result

def test_step1_mlflow():
    """
    1단계 테스트: MLflow 기반 모델 로드 테스트
    
    팀원 제안에 따른 새로운 테스트 함수입니다.
    S3 직접 접근 대신 MLflow 레지스트리를 사용합니다.
    """
    print("🧪 1단계 테스트: MLflow 기반 모델 로드")
    print("=" * 50)
    
    # === 1. 서비스 객체 생성 ===
    print("📦 MoviePredictionService 생성 중...")
    try:
        service = MoviePredictionService()
        print("✅ 서비스 객체 생성 완료")
    except Exception as e:
        print(f"❌ 서비스 객체 생성 실패: {e}")
        print("💡 해결방법:")
        print("   1. MLflow 서버가 실행 중인지 확인")
        print("   2. 환경변수 MLFLOW_TRACKING_URI 설정")
        return False
    
    # === 2. 서비스 상태 확인 ===
    print(f"\n🔍 서비스 상태 확인 중...")
    health_status = service.check_service_health()
    
    print(f"📊 서비스 상태:")
    print(f"   전체 상태: {health_status['status']}")
    print(f"   MLflow 연결: {health_status['components']['mlflow_service']}")
    print(f"   Production 모델: {health_status['components']['production_model']}")
    print(f"   모델 로드: {health_status['components']['model_loaded']}")
    
    # === 3. Production 모델 로드 시도 ===
    if health_status['production_model']['exists']:
        print(f"\n🤖 Production 모델 로드 중...")
        model_result = service.initialize_model()
        
        print(f"\n📊 모델 로드 결과:")
        print(f"   성공 여부: {model_result['success']}")
        print(f"   메시지: {model_result['message']}")
        print(f"   모델 버전: {model_result.get('model_version', 'N/A')}")
        
        if model_result['success']:
            # 모델 상세 정보 출력
            model_info = service.mlflow_service.get_model_info()
            print(f"\n📋 모델 상세 정보:")
            print(f"   모델 타입: {model_info.get('model_type', 'N/A')}")
            print(f"   레지스트리명: {model_info.get('registry_name', 'N/A')}")
            if model_info.get('n_features'):
                print(f"   입력 특성 수: {model_info['n_features']}")
            if model_info.get('is_xgboost'):
                print(f"   XGBoost 모델: ✅")
            
            print(f"\n🎉 1단계 성공!")
            print(f"✅ MLflow 서버 연결: 완료")
            print(f"✅ Production 모델 확인: 완료")
            print(f"✅ 모델 로드: 완료")
            print(f"\n🎯 1단계 완료! 이제 커밋 후 2단계 진행 가능")
            print(f"   → 다음: 2단계 - CSV 데이터 로드 기능 추가")
            return True
        else:
            print(f"\n❌ 1단계 실패: 모델 로드 실패")
            print(f"💡 해결방법:")
            print(f"   1. register_mlflow.py 실행하여 Production 모델 등록")
            print(f"   2. MLflow UI에서 모델 상태 확인")
            return False
    else:
        print(f"\n❌ 1단계 실패: Production 모델이 존재하지 않음")
        print(f"💡 해결방법:")
        print(f"   1. 팀원2의 register_mlflow.py 스크립트 실행")
        print(f"   2. MLflow UI에서 모델이 Production 스테이지에 있는지 확인")
        print(f"   3. 모델 레지스트리 이름 확인: {service.mlflow_service.model_registry_name}")
        return False

def show_mlflow_setup_guide():
    """
    MLflow 설정 가이드 출력
    """
    print("🔧 MLflow 설정 가이드")
    print("=" * 50)
    
    print("📝 1. MLflow 서버 실행:")
    print("   mlflow server --host 0.0.0.0 --port 5001")
    
    print("\n📝 2. 환경변수 설정:")
    print("   export MLFLOW_TRACKING_URI=http://localhost:5001")
    print("   export MODEL_REGISTRY_NAME=MovieRatingXGBoostModel")
    
    print("\n📝 3. Production 모델 등록:")
    print("   팀원2의 register_mlflow.py 스크립트 실행")
    
    print("\n📝 4. MLflow UI 확인:")
    print("   브라우저에서 http://localhost:5001 접속")
    print("   Models 탭에서 Production 스테이지 모델 확인")
    
    print("\n📝 5. 테스트 실행:")
    print("   python movie_service.py")

# 이 파일을 직접 실행할 때만 테스트 함수 실행
if __name__ == "__main__":
    import sys
    
    # 🎯 명령행 인수로 다양한 옵션 제공
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # python movie_service.py test
            test_step1_mlflow()
        elif sys.argv[1] == "help":
            # python movie_service.py help
            show_mlflow_setup_guide()
        else:
            print("사용법:")
            print("  python movie_service.py       # 1단계 MLflow 테스트 실행")
            print("  python movie_service.py test  # 1단계 MLflow 테스트 실행")
            print("  python movie_service.py help  # MLflow 설정 가이드")
    else:
        # python movie_service.py (기본 실행)
        print("🚀 MLflow 기반 1단계 테스트를 시작합니다...")
        print("MLflow 설정이 필요하다면: python movie_service.py help\n")
        test_step1_mlflow()
