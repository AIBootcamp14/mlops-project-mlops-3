import pandas as pd
import logging
import sys
import os

# 경로 설정 - 상위 디렉토리의 services 모듈 import 가능하도록
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 서비스 모듈 import
from services.mlflow_service import MLflowModelService
from services.data_service import DataService

# 로깅 설정
logger = logging.getLogger(__name__)

class SimplePredictionService:
    """
    영화 평점 예측 서비스
    
    MLflowModelService와 DataService를 조합하여
    전체 영화 데이터에 대한 평점 예측을 수행합니다.
    """    
    
    def __init__(self):
        """
        예측 서비스 초기화
        
        필요한 서비스 객체들을 생성하고 상태 변수를 초기화합니다.
        실제 모델과 데이터 로딩은 initialize() 메소드에서 수행됩니다.
        """
        # 서비스 객체 생성
        self.mlflow_service = MLflowModelService()    # 모델 관리 담당
        self.data_service = DataService()             # 데이터 관리 담당
        
        # 예측 결과 저장소
        self.predictions = None                       # 예측 결과가 저장될 딕셔너리
        
        # 서비스 상태 관리
        self.is_ready = False                        # 예측 준비 완료 여부
        
        logger.info("✅ SimplePredictionService 객체 생성 완료")
        
        
    def initialize(self):
        """
        모델과 데이터 로딩
        
        MLflow에서 Production 모델을 로드하고
        전처리된 CSV 데이터를 로드하여 예측 준비를 완료합니다.
        
        Returns:
            dict: 초기화 결과 정보
                - success (bool): 성공 여부
                - message (str): 결과 메시지
                - model_version (str): 로드된 모델 버전
                - data_count (int): 로드된 영화 데이터 개수
        """
        try:
            logger.info("🚀 예측 서비스 초기화 시작...")
            
            # MLflow Production 모델 로드
            logger.info("MLflow Production 모델 로딩 중...")
            model_result = self.mlflow_service.load_production_model()
            
            if not model_result['success']:
                logger.error(f"❌ 모델 로드 실패: {model_result['message']}")
                return {
                    'success': False, 
                    'message': f"모델 로드 실패: {model_result['message']}"
                }
            
            logger.info(f"✅ 모델 로드 성공! 버전: {model_result.get('model_version')}")
                
            # CSV 테스트 데이터 로드
            logger.info("CSV 테스트 데이터 로딩 중...")
            data_result = self.data_service.load_data()
            
            if not data_result['success']:
                logger.error(f"❌ 데이터 로드 실패: {data_result['message']}")
                return {
                    'success': False, 
                    'message': f"데이터 로드 실패: {data_result['message']}"
                }
            
            logger.info(f"✅ 데이터 로드 성공! 영화 수: {data_result['shape'][0]:,}개")
            
            # 예측 준비 완료 상태로 변경
            self.is_ready = True
            logger.info("✅ 예측 서비스 초기화 완료!")
            
            return {
                'success': True,
                'message': '예측 서비스가 성공적으로 초기화되었습니다.',
                'model_version': model_result.get('model_version'),
                'data_count': data_result['shape'][0]
            }
            
        except Exception as e:
            logger.error(f"❌ 초기화 중 예외 발생: {e}")
            self.is_ready = False
            
            return {
                'success': False, 
                'message': f"초기화 실패: {str(e)}"
            }
        
        
    def predict_all(self):
        """
        전체 영화에 대한 평점 예측 수행
        
        로드된 데이터에서 특성을 추출하고 모델을 사용하여
        모든 영화의 평점을 예측합니다.
        
        Returns:
            dict: 예측 결과 정보
                - success (bool): 성공 여부
                - message (str): 결과 메시지
                - sample_count (int): 예측된 영화 개수
        """
        if not self.is_ready:
            return {'success': False, 'message': '서비스가 초기화되지 않았습니다.'}
        
        try:
            logger.info("🤖 전체 데이터 예측 시작...")
            
            # 예측용 데이터 준비
            data = self.data_service.data
            
            # 특성 데이터 추출 (타겟 변수와 ID 제외)
            X = data.drop(['vote_average', 'id'], axis=1)
            
            # 영화 ID는 결과 매핑용으로 별도 저장
            movie_ids = data['id'].tolist()
            
            # 모델을 사용한 예측 수행
            predictions = self.mlflow_service.model.predict(X)
            
            # 예측 결과 저장
            self.predictions = {
                'predictions': predictions.tolist(),  # numpy array를 list로 변환
                'movie_ids': movie_ids,
                'sample_count': len(predictions)
            }
            
            logger.info(f"✅ 예측 완료! {len(predictions)}개 영화")
            
            return {
                'success': True,
                'message': f'{len(predictions)}개 영화에 대한 예측이 완료되었습니다.',
                'sample_count': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"❌ 예측 실패: {e}")
            return {'success': False, 'message': f"예측 실패: {str(e)}"}
        
        
    def get_predictions(self):
        """
        전체 예측 결과 조회
        
        수행된 예측 결과를 API 응답 형태로 반환합니다.
        
        Returns:
            dict: 예측 결과 데이터
                - available (bool): 예측 결과 존재 여부
                - sample_count (int): 전체 영화 수
                - results (list): 영화별 예측 결과 리스트
        """
        if not self.predictions:
            return {'available': False, 'message': '예측 결과가 없습니다.'}
        
        return {
            'available': True,
            'sample_count': self.predictions['sample_count'],
            'results': [
                {
                    'movie_id': movie_id,
                    'predicted_rating': pred
                }
                for movie_id, pred in zip(
                    self.predictions['movie_ids'], 
                    self.predictions['predictions']
                )
            ]
        }
        
        
    def get_top_movies(self, top_n=10):
        """
        예측 평점 상위 영화 조회
        
        예측된 평점을 기준으로 상위 N개 영화를 반환합니다.
        
        Args:
            top_n (int): 조회할 상위 영화 개수 (기본값: 10)
            
        Returns:
            dict: 상위 영화 리스트
                - available (bool): 데이터 존재 여부
                - top_movies (list): 상위 영화 정보 리스트
                - total_count (int): 전체 영화 수
        """
        if not self.predictions:
            return {'available': False, 'message': '예측 결과가 없습니다.'}
        
        # 예측 평점 기준으로 정렬
        predictions = self.predictions['predictions']
        movie_ids = self.predictions['movie_ids']
        
        # 평점 높은 순으로 정렬하여 상위 N개 추출
        sorted_data = sorted(
            zip(movie_ids, predictions), 
            key=lambda x: x[1],  # 예측 평점 기준
            reverse=True         # 내림차순
        )[:top_n]
        
        # 결과 포맷팅
        top_movies = []
        for rank, (movie_id, predicted_rating) in enumerate(sorted_data, 1):
            top_movies.append({
                'rank': rank,
                'movie_id': movie_id,
                'predicted_rating': round(predicted_rating, 2)
            })
        
        return {
            'available': True,
            'top_movies': top_movies,
            'total_count': len(predictions)
        }
        
        
    def get_status(self):
        """
        예측 서비스 상태 조회
        
        현재 서비스의 전반적인 상태를 확인합니다.
        
        Returns:
            dict: 서비스 상태 정보
                - service_ready (bool): 서비스 준비 완료 여부
                - model_loaded (bool): 모델 로드 상태
                - data_loaded (bool): 데이터 로드 상태
                - predictions_available (bool): 예측 결과 존재 여부
                - sample_count (int): 예측된 영화 수
        """
        return {
            'service_ready': self.is_ready,
            'model_loaded': self.mlflow_service.is_model_loaded,
            'data_loaded': self.data_service.is_data_loaded,
            'predictions_available': self.predictions is not None,
            'sample_count': self.predictions['sample_count'] if self.predictions else 0
        }

# =============================================================================
# 테스트 함수
# =============================================================================

def test_prediction_service():
    """예측 서비스 기본 테스트"""
    print("🧪 예측 서비스 테스트")
    print("=" * 40)
    
    # 서비스 생성
    print("📦 서비스 생성 중...")
    service = SimplePredictionService()
    print("✅ 객체 생성 완료")
    
    # 초기화
    print("\n🚀 서비스 초기화 중...")
    init_result = service.initialize()
    print(f"초기화: {'✅ 성공' if init_result['success'] else '❌ 실패'}")
    print(f"메시지: {init_result['message']}")
    
    if init_result['success']:
        print(f"모델 버전: {init_result.get('model_version', 'N/A')}")
        print(f"영화 데이터: {init_result.get('data_count', 0):,}개")
        
        # 서비스 상태
        print(f"\n📊 서비스 상태:")
        status = service.get_status()
        print(f"   서비스 준비: {'✅' if status['service_ready'] else '❌'}")
        print(f"   모델 로드: {'✅' if status['model_loaded'] else '❌'}")
        print(f"   데이터 로드: {'✅' if status['data_loaded'] else '❌'}")
        
        print(f"\n🎉 테스트 성공!")
        print(f"📋 다음: predict_all() 메소드 테스트")
        return True
    else:
        print(f"\n❌ 초기화 실패")
        print(f"💡 확인사항:")
        print(f"   - MLflow 서버 실행 여부")
        print(f"   - Production 모델 등록 여부")
        print(f"   - CSV 파일 존재 여부")
        return False

def debug_environment():
    """환경 설정 디버깅"""
    print("🔍 환경 설정 확인")
    print("=" * 40)
    
    # 현재 경로
    print(f"현재 디렉토리: {os.getcwd()}")
    
    # Python 경로
    print(f"Python 모듈 경로:")
    for path in sys.path[:5]:  # 상위 5개만 표시
        print(f"   {path}")
    
    # 파일 존재 확인
    service = DataService()
    print(f"CSV 파일 경로: {service.csv_path}")
    print(f"파일 존재: {'✅' if os.path.exists(service.csv_path) else '❌'}")

# 실행 부분
if __name__ == "__main__":
    import sys
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "debug":
            debug_environment()
        elif sys.argv[1] == "test":
            test_prediction_service()
        else:
            print("사용법:")
            print("  python prediction_service.py       # 기본 테스트")
            print("  python prediction_service.py test  # 기본 테스트")
            print("  python prediction_service.py debug # 환경 디버깅")
    else:
        test_prediction_service()