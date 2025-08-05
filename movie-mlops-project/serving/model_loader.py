# import pickle
# import boto3
# import logging
# import os
# import io
# from typing import Optional, Dict, Any
# from datetime import datetime
# from botocore.exceptions import ClientError, NoCredentialsError

# # 로거 설정
# logger = logging.getLogger(__name__)

# class MovieRatingPredictor:
#     """
#     영화 평점 예측 모델 관리 클래스
    
#     현재 단계: S3에서 XGBoost 모델 로드만 담당
#     """
    
#     def __init__(self):
#         """
#         예측기 초기화
#         S3 설정 값들을 환경변수에서 읽어오거나 기본값으로 설정
#         클래스가 처음 생성될 때 한번만 실행
#         필요한 변수들을 초기 설정
#         S3 클라이언트 초기화 시도
#         """
#         # 모델 관련 상태 변수들
#         self.model = None               # XGBoost 모델이 저장될 공간 (처음엔 비어있음)
#         self.model_version = None       # 모델 버전 정보
#         self.is_loaded = False          # 모델이 로드되었는가 확인
#         self.load_timestamp = None      # 언제 모델을 로드했는지 시간 기록
        
#         # S3 설정 - 현재 환경에 맞춘 실제 값들
#         self.bucket_name = os.getenv("S3_BUCKET_NAME", "mlopsproject-3")        # S3 버킷명
#         self.aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")     # AWS 리전 (서울)
#         self.model_key = os.getenv("S3_MODEL_KEY", "models/xgb_md6_eta0_3.pkl") # 모델 파일 경로
#         self.download_timeout = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "30"))  # 다운로드 타임아웃
        
#         # S3 클라이언트 (초기화는 별도 메서드에서)
#         self.s3_client = None           # S3 클라이언트 객체 저장 공간
        
#         # S3 클라이언트 초기화 시도
#         self._init_s3_client()
        
    
#     def _init_s3_client(self) -> bool:
#         """
#         S3 클라이언트를 초기화
#         AWS S3에 연결하기 위한 클라이언트 객체를 생성
#         여러 가지 인증 방법을 시도
#         성공 시 self.s3_client에 저장
#         __init__()함수에서 자동으로 호출 됨
        
#         AWS 인증 방법:
#         1. 환경변수 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
#         2. IAM 역할 (EC2에서 실행시)
#         3. ~/.aws/credentials 파일
#         4. AWS CLI 설정
        
#         Returns:
#             bool: 초기화 성공 여부
#         """
#         try:
#             logger.info("S3 클라이언트 초기화 시작...")
            
#             # 환경변수에서 AWS 인증 정보 확인
#             aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
#             aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
#             if aws_access_key and aws_secret_key:
#                 # 방법 1: 명시적 인증 정보 사용
#                 logger.info("환경변수에서 AWS 인증정보 발견, 명시적 인증 사용")
#                 self.s3_client = boto3.client(
#                     's3',
#                     aws_access_key_id=aws_access_key,
#                     aws_secret_access_key=aws_secret_key,
#                     region_name=self.aws_region
#                 )
#             else:
#                 # 방법 2: 기본 인증 방식 사용 (IAM 역할, 프로파일 등)
#                 logger.info("환경변수에 인증정보 없음, 기본 인증 방식 사용")
#                 self.s3_client = boto3.client('s3', region_name=self.aws_region)
            
#             logger.info(f"✅ S3 클라이언트 초기화 완료 (리전: {self.aws_region})")
#             return True
        
#         except Exception as e:
#             logger.error(f"❌ S3 클라이언트 초기화 실패: {str(e)}")
#             self.s3_client = None
#             return False
        
        
#     def check_s3_connection(self) -> Dict[str, Any]:
#         """
#         S3 연결 상태를 확인
#         버킷에 실제로 접근 가능한지 테스트
        
#         Returns:
#             Dict: 연결 상태 정보
#             {
#                 'connected': bool,      # 연결 성공 여부
#                 'bucket_name': str,     # 버킷명 (성공시)
#                 'region': str,          # 리전 (성공시)
#                 'error': str            # 오류 메시지 (실패시)
#             }
#         """
#         logger.info(f"S3 연결 상태 확인 중... (버킷: {self.bucket_name})")
        
#         if not self.s3_client:
#             return {
#                 'connected': False,
#                 'error': 'S3 클라이언트가 초기화되지 않았습니다.'
#             }
        
#         try:
#             # 버킷 헤더 정보 조회로 접근 권한 확인
#             self.s3_client.head_bucket(Bucket=self.bucket_name)
            
#             logger.info(f"✅ S3 버킷 연결 성공: {self.bucket_name}")
#             return {
#                 'connected': True,
#                 'bucket_name': self.bucket_name,
#                 'region': self.aws_region
#             }
            
#         except ClientError as e:
#             error_code = e.response['Error']['Code']
#             if error_code == 'NoSuchBucket':
#                 error_msg = f"버킷 '{self.bucket_name}'을 찾을 수 없습니다."
#             elif error_code == 'AccessDenied':
#                 error_msg = f"버킷 '{self.bucket_name}'에 대한 접근 권한이 없습니다."
#             else:
#                 error_msg = f"S3 오류: {error_code}"
                
#             logger.error(f"❌ S3 연결 실패: {error_msg}")
#             return {
#                 'connected': False,
#                 'error': error_msg
#             }
#         except Exception as e:
#             error_msg = f"예상치 못한 오류: {str(e)}"
#             logger.error(f"❌ S3 연결 확인 중 오류: {error_msg}")
#             return {
#                 'connected': False,
#                 'error': error_msg
#             }
        
    
#     def _download_model_from_s3(self, bucket_name: str, model_key: str) -> Optional[bytes]:
#         """
#         S3에서 모델 파일을 다운로드
#         load_model() 함수 내에서 호출
        
#         Args:
#             bucket_name: S3 버킷명
#             model_key: 모델 파일의 S3 키 (경로)
            
#         Returns:
#             Optional[bytes]: 다운로드된 모델 파일의 바이트 데이터, 실패시 None
#         """
#         if not self.s3_client:
#             logger.error("S3 클라이언트가 초기화되지 않았습니다.")
#             return None
        
#         try:
#             logger.info(f"📥 S3에서 모델 다운로드 시작...")
#             logger.info(f"   버킷: {bucket_name}")
#             logger.info(f"   키: {model_key}")
            
#             # S3 GetObject API 호출로 파일 다운로드
#             response = self.s3_client.get_object(
#                 Bucket=bucket_name,
#                 Key=model_key
#             )
            
#             # 스트림에서 모든 바이트 데이터 읽기
#             model_data = response['Body'].read()
            
#             # 다운로드 완료 정보 로깅
#             file_size_bytes = len(model_data)
#             file_size_mb = file_size_bytes / (1024 * 1024)
#             logger.info(f"✅ 모델 다운로드 완료!")
#             logger.info(f"   파일 크기: {file_size_mb:.2f}MB ({file_size_bytes:,} bytes)")
            
#             return model_data
            
#         except ClientError as e:
#             # AWS S3 관련 오류들 처리
#             error_code = e.response['Error']['Code']
#             if error_code == 'NoSuchKey':
#                 logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_key}")
#             elif error_code == 'NoSuchBucket':
#                 logger.error(f"❌ S3 버킷을 찾을 수 없습니다: {bucket_name}")
#             elif error_code == 'AccessDenied':
#                 logger.error("❌ S3 접근 권한이 없습니다. IAM 정책을 확인하세요.")
#             else:
#                 logger.error(f"❌ S3 API 오류: {error_code} - {str(e)}")
#             return None
            
#         except NoCredentialsError:
#             logger.error("❌ AWS 인증 정보가 없습니다. 다음을 확인하세요:")
#             logger.error("   1. 환경변수: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
#             logger.error("   2. IAM 역할 (EC2에서 실행시)")
#             logger.error("   3. ~/.aws/credentials 파일")
#             return None
            
#         except Exception as e:
#             logger.error(f"❌ 모델 다운로드 중 예상치 못한 오류: {str(e)}")
#             return None
        
        
#     def load_model(self, 
#                    bucket_name: Optional[str] = None, 
#                    model_key: Optional[str] = None) -> bool:
#         """
#         S3에서 XGBoost pickle 모델을 로드
#         전체 모델 로드 프로세스를 총괄하는 메인 함수
        
#         Args:
#             bucket_name: S3 버킷명 (None인 경우 기본값 사용)
#             model_key: 모델 파일 키 (None인 경우 기본값 사용)
            
#         Returns:
#             bool: 모델 로드 성공 여부
#         """
#         try:
#             # 파라미터 기본값 설정 (None인 경우 인스턴스 설정값 사용)
#             bucket = bucket_name or self.bucket_name    # 기본 값: mlopsproject-3
#             key = model_key or self.model_key           # 기본 값: models/xgb_md6_et0_3.pkl
            
#             logger.info("🚀 XGBoost 모델 로드 프로세스 시작")
#             logger.info(f"   대상: s3://{bucket}/{key}")
            
#             # 1단계: S3에서 모델 파일 다운로드
#             model_data = self._download_model_from_s3(bucket, key)  # s3에 있는 모델 데이터 받아오기
#             if model_data is None:
#                 logger.error("❌ 모델 다운로드 실패로 인해 로드 중단")
#                 return False
            
#             # 2단계: 바이트 데이터를 파일 객체로 변환
#             logger.info("🔄 모델 데이터 변환 중...")
#             model_buffer = io.BytesIO(model_data)   # 메모리에서 파일처럼 사용할 수 있게 변환
            
#             # 3단계: pickle로 모델 역직렬화
#             logger.info("📦 pickle로 모델 역직렬화 중...")
#             self.model = pickle.load(model_buffer)  # 바이트 데이터 -> XGBoost 모델 객체
            
#             # 4단계: 로드된 모델 정보 확인
#             model_type = type(self.model).__name__
#             logger.info(f"✅ 모델 타입 확인: {model_type}")
            
#             # 5단계: 모델 버전 추출 (파일명에서)
#             # 예: "xgb_md6_eta0_3.pkl" -> "md6_eta0_3"
#             try:
#                 filename = key.split('/')[-1]  # 경로에서 파일명만 추출
#                 if filename.startswith('xgb_') and filename.endswith('.pkl'):
#                     version_part = filename[4:-4]  # "xgb_"와 ".pkl" 제거
#                     self.model_version = f"XGBoost_{version_part}"
#                 else:
#                     self.model_version = "XGBoost_unknown"
#                 logger.info(f"📋 모델 버전: {self.model_version}")
#             except Exception as ve:
#                 logger.warning(f"⚠️ 모델 버전 추출 실패: {str(ve)}")
#                 self.model_version = "XGBoost_unknown"
            
#             # 6단계: 상태 업데이트
#             self.is_loaded = True                   # 로드 완료 표시
#             self.load_timestamp = datetime.now()    # 현재 시간 기록
            
#             logger.info("🎉 XGBoost 모델 로드 완료!")
#             logger.info(f"   로드 시각: {self.load_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
#             return True
            
#         except pickle.UnpicklingError as e:
#             logger.error("❌ Pickle 역직렬화 실패:")
#             logger.error(f"   파일이 손상되었거나 올바른 pickle 파일이 아닐 수 있습니다.")
#             logger.error(f"   상세 오류: {str(e)}")
#             return False
            
#         except Exception as e:
#             logger.error(f"❌ 모델 로드 중 예상치 못한 오류: {str(e)}")
#             # 실패시 상태 초기화
#             self.model = None
#             self.is_loaded = False
#             return False
        
        
#     def get_model_info(self) -> Dict[str, Any]:
#         """
#         현재 로드된 모델의 상세 정보를 반환
#         상태 확인, 디버깅, 모니터링에 사용
        
#         Returns:
#             Dict: 모델 정보
#         """
#         base_info = {
#             'is_loaded': self.is_loaded,                            # 모델이 로드 되었는지
#             'model_version': self.model_version,                    # 모델 버전
#             'model_type': 'XGBoost' if self.is_loaded else None,    # 모델 타입
#             'bucket_name': self.bucket_name,                        # S3 버킷 명
#             'model_key': self.model_key,                            # 모델 파일 경로
#             'load_timestamp': self.load_timestamp.isoformat() if self.load_timestamp else None, # 로드한 시간
#             'aws_region': self.aws_region                           # AWS 리전
#         }
        
#         # 모델이 로드된 경우 추가 정보 제공
#         if self.is_loaded and self.model:
#             try:
#                 # XGBoost 모델의 세부 정보 (가능한 경우)
#                 if hasattr(self.model, 'get_booster'):              # XGBoost 모델인지 확인
#                     booster = self.model.get_booster()
#                     base_info.update({
#                         'num_trees': booster.num_boosted_rounds(),  # XGBoost 트리 개수
#                         'num_features': booster.num_features()      # 입력 특성 개수
#                     })
#                     logger.info(f"XGBoost 모델 세부 정보 - 트리 수: {base_info['num_trees']}, 특성 수: {base_info['num_features']}")
                    
#             except Exception as e:
#                 logger.warning(f"XGBoost 세부 정보 추출 실패: {str(e)}")
        
#         return base_info
    

# # 전역 예측기 인스턴스 (싱글톤 패턴)
# # 서버에서 하나의 인스턴스만 생성해서 메모리 효율성 확보
# predictor = MovieRatingPredictor()


# def get_predictor() -> MovieRatingPredictor:
#     """
#     전역 예측기 인스턴스를 반환합니다.
#     싱글톤 패턴을 구현
#     어디서든 같은 예측기 인스턴스를 사용하게 함
    
#     Returns:
#         MovieRatingPredictor: 예측기 인스턴스
#     """
#     return predictor