import os
import logging
import mlflow
import mlflow.xgboost

# .env 파일 로드
try:
    from dotenv import load_dotenv
    
    possible_paths = [
        ".env",
        "../.env", 
        "../../.env"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            load_dotenv(dotenv_path=path, override=True)
            logging.info(f"📁 .env 파일 로드: {path}")
            break
    
except ImportError:
    logging.warning("⚠️ python-dotenv가 설치되지 않음")

# 로깅 설정
logger = logging.getLogger(__name__)

class MLflowModelService:
    """
    MLflow 모델 레지스트리 기반 모델 관리 서비스
    """
    
    def __init__(self):
        """MLflow 모델 서비스 초기화"""
        
        # AWS 자격 증명 명시적 설정
        self._setup_aws_credentials()
        
        # MLflow 관련 설정
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        self.model_registry_name = os.getenv("MODEL_REGISTRY_NAME", "MovieRatingXGBoostModel")
        
        # 모델 상태 관리
        self.model = None
        self.is_model_loaded = False
        self.model_version = None
        self.model_uri = None
        
        # MLflow 설정
        self._setup_mlflow()
        
        logger.info("✅ MLflowModelService 초기화 완료")
        
    def _setup_aws_credentials(self):
        """AWS 자격 증명 명시적 설정"""
        try:
            # .env 파일에서 AWS 자격 증명 읽기
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
            
            if aws_access_key and aws_secret_key:
                # 환경변수로 명시적 설정 (MLflow가 boto3 사용할 때 참조)
                os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
                os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
                os.environ['AWS_DEFAULT_REGION'] = aws_region
                
                logger.info("✅ AWS 자격 증명 설정 완료")
                logger.info(f"   Access Key: {aws_access_key[:4]}***{aws_access_key[-4:]}")
                logger.info(f"   Region: {aws_region}")
            else:
                logger.warning("⚠️ AWS 자격 증명이 .env 파일에 없습니다")
                
        except Exception as e:
            logger.error(f"❌ AWS 자격 증명 설정 실패: {e}")
            
    def _setup_mlflow(self):
        """MLflow Tracking URI 설정"""
        try:
            logger.info(f"🔗 MLflow 서버 연결 중: {self.mlflow_tracking_uri}")
            
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            logger.info("✅ MLflow 서버 연결 완료")
            logger.info(f"   📋 모델 레지스트리명: {self.model_registry_name}")
            
        except Exception as e:
            logger.error(f"❌ MLflow 서버 연결 실패: {e}")
            raise
            
    def check_mlflow_connection(self):
        """MLflow 서버 연결 상태 확인"""
        try:
            logger.info("🔍 MLflow 서버 연결 확인 중...")
            
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
        """Production 스테이지 모델 존재 여부 확인"""
        try:
            logger.info(f"🔍 Production 모델 존재 확인: {self.model_registry_name}")
            
            client = mlflow.MlflowClient()
            
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
        """Production 스테이지 모델 로드"""
        
        if self.is_model_loaded:
            logger.info("모델이 이미 로드되었습니다.")
            return {
                'success': True,
                'message': '모델이 이미 로드되어 있습니다.',
                'model_loaded': True,
                'model_version': self.model_version
            }
        
        # MLflow 연결 상태 확인
        connection_check = self.check_mlflow_connection()
        if not connection_check['success']:
            return {
                'success': False,
                'message': f"MLflow 연결 실패: {connection_check['message']}",
                'model_loaded': False,
                'model_version': None
            }
        
        # Production 모델 존재 확인
        model_check = self.check_production_model_exists()
        if not model_check['exists']:
            return {
                'success': False,
                'message': f"Production 모델 없음: {model_check['message']}",
                'model_loaded': False,
                'model_version': None
            }
        
        try:
            # 모델 URI 설정
            self.model_uri = f"models:/{self.model_registry_name}/Production"
            
            logger.info(f"🤖 Production 모델 로딩 중...")
            logger.info(f"   📍 모델 URI: {self.model_uri}")
            logger.info(f"   🏷️ 모델 버전: {model_check['version']}")
            
            # MLflow로 모델 로드
            self.model = mlflow.xgboost.load_model(self.model_uri)
            
            # 상태 업데이트
            self.is_model_loaded = True
            self.model_version = model_check['version']
            
            logger.info("✅ Production 모델 로드 완료!")
            logger.info(f"   🏷️ 모델 타입: {type(self.model)}")
            
            # XGBoost 모델 정보 확인
            try:
                if hasattr(self.model, 'n_features_in_'):
                    logger.info(f"   📊 입력 특성 수: {self.model.n_features_in_}")
                if hasattr(self.model, 'get_booster'):
                    logger.info(f"   🚀 XGBoost 모델 확인됨")
            except:
                pass
            
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
        """현재 로드된 모델 정보 조회"""
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