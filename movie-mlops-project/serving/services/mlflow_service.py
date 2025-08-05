import os
import logging
import mlflow
import mlflow.xgboost

# 로깅 설정
logger = logging.getLogger(__name__)

class MLflowModelService:
    """
    MLflow 모델 레지스트리 기반 모델 관리 서비스
    
    ✅ 주요 기능:
    - Production 스테이지 모델 자동 로드
    - 모델 버전 관리 자동화
    - MLflow Tracking Server 연결
    
    """
    
    def __init__(self):
        """
        MLflow 모델 서비스 초기화
        환경변수에서 MLflow 설정을 읽어와 초기화
        """
        # === MLflow 관련 설정 ===
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        self.model_registry_name = os.getenv("MODEL_REGISTRY_NAME", "MovieRatingXGBoostModel")
        
        # === 모델 상태 관리 ===
        self.model = None  # XGBoost 모델 객체
        self.is_model_loaded = False  # 모델 로드 완료 여부
        self.model_version = None  # 로드된 모델 버전
        self.model_uri = None  # 모델 URI
        
        # === MLflow 설정 ===
        self._setup_mlflow()
        
        logger.info("✅ MLflowModelService 초기화 완료")
        
    def _setup_mlflow(self):
        """
        MLflow Tracking URI 설정
        MLflow 서버와의 연결을 설정
        """
        try:
            logger.info(f"🔗 MLflow 서버 연결 중: {self.mlflow_tracking_uri}")
            
            # MLflow Tracking URI 설정
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            logger.info("✅ MLflow 서버 연결 완료")
            logger.info(f"   📋 모델 레지스트리명: {self.model_registry_name}")
            
        except Exception as e:
            logger.error(f"❌ MLflow 서버 연결 실패: {e}")
            logger.error("💡 MLflow 서버가 실행 중인지 확인하세요.")
            raise
            
    def check_mlflow_connection(self):
        """
        MLflow 서버 연결 상태 확인
        MLflow 서버에 실제로 연결 가능한지 테스트
        
        Returns:
            dict: 연결 상태 정보
        """
        try:
            logger.info("🔍 MLflow 서버 연결 확인 중...")
            
            # MLflow 서버에서 실험 목록 가져오기 (연결 테스트)
            experiments = mlflow.search_experiments()
            
            logger.info("✅ MLflow 서버 연결 성공!")
            logger.info(f"   📊 등록된 실험 수: {len(experiments)}")
            
            return {
                'success': True,
                'message': 'MLflow 서버에 정상적으로 연결되었습니다.',
                'tracking_uri': self.mlflow_tracking_uri,
                'experiments_count': len(experiments)
            }
            
        except Exception as e:
            logger.error(f"❌ MLflow 서버 연결 실패: {e}")
            
            return {
                'success': False,
                'message': f'MLflow 서버 연결 실패: {str(e)}',
                'tracking_uri': self.mlflow_tracking_uri,
                'experiments_count': 0
            }
            
    def check_production_model_exists(self):
        """
        Production 스테이지 모델 존재 여부 확인
        모델 레지스트리에서 Production 스테이지의 모델이 있는지 확인
        
        Returns:
            dict: Production 모델 존재 여부 정보
        """
        try:
            logger.info(f"🔍 Production 모델 존재 확인: {self.model_registry_name}")
            
            # MLflow 클라이언트를 통해 모델 정보 조회
            client = mlflow.MlflowClient()
            
            # Production 스테이지의 모델 버전 조회
            production_versions = client.get_latest_versions(
                name=self.model_registry_name,
                stages=["Production"]
            )
            
            if production_versions:
                latest_version = production_versions[0]
                
                logger.info("✅ Production 모델 존재 확인!")
                logger.info(f"   🏷️ 모델 버전: {latest_version.version}")
                logger.info(f"   📅 등록 시간: {latest_version.creation_timestamp}")
                
                return {
                    'exists': True,
                    'message': 'Production 모델이 존재합니다.',
                    'version': latest_version.version,
                    'creation_time': latest_version.creation_timestamp,
                    'model_uri': f"models:/{self.model_registry_name}/Production"
                }
            else:
                logger.warning("⚠️ Production 스테이지에 모델이 없습니다.")
                
                return {
                    'exists': False,
                    'message': 'Production 스테이지에 등록된 모델이 없습니다.',
                    'version': None,
                    'creation_time': None,
                    'model_uri': None
                }
                
        except Exception as e:
            logger.error(f"❌ Production 모델 확인 실패: {e}")
            
            return {
                'exists': False,
                'message': f'모델 확인 실패: {str(e)}',
                'version': None,
                'creation_time': None,
                'model_uri': None
            }
            
    def load_production_model(self):
        """
        MLflow 모델 레지스트리에서 Production 스테이지의 모델을 로드
        
        🎯 팀원 제안의 핵심 기능:
        - S3 경로나 버킷 이름 불필요
        - Production 스테이지 모델 자동 로드
        - 모델 버전 변경 시 코드 수정 불필요
        
        Returns:
            dict: 모델 로드 결과 정보
        """
        
        # 🔄 이미 로드되어 있다면 스킵
        if self.is_model_loaded:
            logger.info("모델이 이미 로드되었습니다.")
            return {
                'success': True,
                'message': '모델이 이미 로드되어 있습니다.',
                'model_loaded': True,
                'model_version': self.model_version
            }
        
        # === 1. MLflow 연결 상태 확인 ===
        connection_check = self.check_mlflow_connection()
        if not connection_check['success']:
            return {
                'success': False,
                'message': f"MLflow 연결 실패: {connection_check['message']}",
                'model_loaded': False,
                'model_version': None
            }
        
        # === 2. Production 모델 존재 확인 ===
        model_check = self.check_production_model_exists()
        if not model_check['exists']:
            return {
                'success': False,
                'message': f"Production 모델 없음: {model_check['message']}",
                'model_loaded': False,
                'model_version': None
            }
        
        try:
            # === 3. 모델 URI 설정 ===
            self.model_uri = f"models:/{self.model_registry_name}/Production"
            
            logger.info(f"🤖 Production 모델 로딩 중...")
            logger.info(f"   📍 모델 URI: {self.model_uri}")
            logger.info(f"   🏷️ 모델 버전: {model_check['version']}")
            
            # === 4. 🎯 핵심: MLflow로 모델 로드 (한 줄로 끝!) ===
            # 이 한 줄이 S3에서 모델을 자동으로 찾아 로드합니다.
            self.model = mlflow.xgboost.load_model(self.model_uri)
            
            # === 5. 상태 업데이트 ===
            self.is_model_loaded = True
            self.model_version = model_check['version']
            
            logger.info("✅ Production 모델 로드 완료!")
            logger.info(f"   🏷️ 모델 타입: {type(self.model)}")
            
            # 🔍 XGBoost 모델 정보 확인
            try:
                if hasattr(self.model, 'n_features_in_'):
                    logger.info(f"   📊 입력 특성 수: {self.model.n_features_in_}")
                if hasattr(self.model, 'get_booster'):
                    logger.info(f"   🚀 XGBoost 모델 확인됨")
            except:
                pass  # 추가 정보 확인 실패해도 무시
            
            return {
                'success': True,
                'message': 'Production 모델 로드가 성공적으로 완료되었습니다.',
                'model_loaded': True,
                'model_version': self.model_version
            }
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            
            # 실패 시 상태 초기화
            self.is_model_loaded = False
            self.model = None
            self.model_version = None
            
            return {
                'success': False,
                'message': f'MLflow 모델 로드 실패: {str(e)}',
                'model_loaded': False,
                'model_version': None
            }
            
    def get_model_info(self):
        """
        현재 로드된 모델 정보 조회
        
        Returns:
            dict: 모델 정보
        """
        if not self.is_model_loaded:
            return {
                'model_loaded': False,
                'message': '모델이 로드되지 않았습니다.'
            }
        
        info = {
            'model_loaded': True,
            'model_version': self.model_version,
            'model_uri': self.model_uri,
            'model_type': str(type(self.model)),
            'registry_name': self.model_registry_name
        }
        
        # XGBoost 모델 추가 정보
        try:
            if hasattr(self.model, 'n_features_in_'):
                info['n_features'] = self.model.n_features_in_
            if hasattr(self.model, 'get_booster'):
                info['is_xgboost'] = True
        except:
            pass
        
        return info