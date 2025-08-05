import boto3      # AWS S3 접근용 라이브러리
import logging    # 로그 출력용 라이브러리
import os         # 환경변수 및 파일 경로 처리용

# 로깅 설정 - 콘솔에 정보를 출력하도록 설정 (먼저 설정!)
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨 이상만 출력 (DEBUG는 출력 안함)
    format="%(asctime)s - %(levelname)s - %(message)s"  # 시간 - 레벨 - 메시지 형식
)
logger = logging.getLogger(__name__)  # 현재 모듈명으로 로거 생성

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
            logger.info(f"📁 .env 파일 로드 완료: {path}")
            env_loaded = True
            break
    
    if not env_loaded:
        logger.warning("⚠️ .env 파일을 찾을 수 없음")
    
    # 🔍 실제로 AWS 키가 로드되었는지 확인
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    if aws_key:
        # 보안을 위해 키의 일부만 표시
        masked_key = aws_key[:4] + "*" * (len(aws_key) - 8) + aws_key[-4:]
        logger.info(f"🔑 AWS Access Key 확인: {masked_key}")
    else:
        logger.warning("⚠️ AWS_ACCESS_KEY_ID 환경변수가 설정되지 않음")
    
except ImportError:
    logger.warning("⚠️ python-dotenv가 설치되지 않음. 환경변수를 직접 설정하세요.")
    logger.warning("💡 설치 방법: pip install python-dotenv")
except Exception as e:
    logger.warning(f"⚠️ .env 파일 로드 실패: {e}")
    logger.warning("💡 환경변수를 직접 설정하거나 .env 파일 경로를 확인하세요.")

class MoviePredictionService:
    """
    영화 평점 예측 서비스 클래스
    
    - 기본 설정값 초기화
    - S3 클라이언트 연결 설정
    - S3 연결 상태 확인
    """
    def __init__(self):
        """
        객체가 생성될 때 한 번만 실행되며, 필요한 설정값들을 저장
        아직 실제 작업(다운로드, 예측 등)은 하지 않고 설정만 준비
        """
        # S3 관련 설정값들 (AWS 클라우드 스토리지)
        self.bucket_name = "mlopsproject-3"  # 모델이 저장된 S3 버킷 이름
        self.model_key = "models/xgb_md6_eta0_3.pkl"  # 버킷 내 모델 파일 경로
        self.aws_region = "ap-northeast-2"  # AWS 서울 리전
        
        # 로컬 파일 경로
        self.test_csv_path = "../preprocessing/result/tmdb_test.csv"
        
        # 서비스 상태를  나타내는 변수
        self.model = None # XGBoost 모델 객체가 저장될 공간
        self.is_model_loaded = False  # 모델 로드 완료 여부 (True/False)
        self.test_data = None  # CSV 데이터가 저장될 공간
        self.predictions = []  # 예측 결과들이 저장될 리스트
        
        # S3 클라이언트 (AWS와 통신하는 객체)
        self.s3_client = None
        
        # S3 클라이언트 설정 실행
        self._setup_s3()
        
        logger.info("✅ MoviePredictionService 초기화 완료 (1단계)")
        
        
    def _setup_s3(self):
        """
        S3 클라이언트 설정 함수
        
        AWS S3와 통신할 수 있는 클라이언트 객체를 생성
        환경변수나 AWS 설정 파일에서 인증 정보를 자동으로 찾아 사용
        
        인증 정보를 찾는 순서:
        1. 환경변수 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        2. ~/.aws/credentials 파일
        3. IAM 역할 (EC2에서 실행시)
        """
        try:
            logger.info("🔗 S3 클라이언트 설정 중...")
            
            # boto3.client()로 S3 클라이언트 생성
            # region_name: 데이터센터 위치 지정 (서울 리전)
            self.s3_client = boto3.client('s3', region_name=self.aws_region)
            
            logger.info(f"✅ S3 클라이언트 설정 완료 (리전: {self.aws_region})")
            
        except Exception as e:
            # 인증 정보가 없거나 잘못된 경우 오류 발생
            logger.error(f"❌ S3 클라이언트 설정 실패: {e}")
            logger.error("💡 AWS 인증 정보를 확인하세요:")
            logger.error("   - 환경변수: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            logger.error("   - 파일: ~/.aws/credentials")
            logger.error("   - IAM 역할 (EC2에서 실행시)")
            
            
    def check_s3_connection(self):
        """
        S3 연결 상태 확인 함수 (1단계)
        
        실제로 S3 버킷에 접근 가능한지 테스트
        클라이언트 생성은 성공해도 실제 접근 권한이 없을 수 있기 때문에
        head_bucket() API를 호출해서 실제 접근 가능 여부를 확인
        
        Returns:
            dict: 연결 상태 정보
            {
                'success': True/False,
                'message': '상태 메시지',
                'bucket_name': '버킷명'
            }
        """
        if not self.s3_client:
            return {
                'success': False,
                'message': 'S3 클라이언트가 설정되지 않았습니다.',
                'bucket_name': None
            }
            
        try:
            logger.info(f"🔍 S3 버킷 연결 확인 중: {self.bucket_name}")
            
            # head_bucket() - 버킷 정보를 가져오는 API 호출
            # 이 호출이 성공하면 버킷에 접근 권한이 있다는 의미
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            logger.info(f"✅ S3 버킷 연결 성공!")
            
            return {
                'success': True,
                'message': 'S3 버킷에 정상적으로 연결되었습니다.',
                'bucket_name': self.bucket_name
            }
            
        except Exception as e:
            logger.error(f"❌ S3 버킷 연결 실패: {e}")
            
            # 오류 종류별 상세 메시지
            error_message = str(e)
            if "NoSuchBucket" in error_message:
                detailed_message = f"버킷 '{self.bucket_name}'을 찾을 수 없습니다."
            elif "AccessDenied" in error_message:
                detailed_message = f"버킷 '{self.bucket_name}'에 대한 접근 권한이 없습니다."
            else:
                detailed_message = f"알 수 없는 오류: {error_message}"
            
            return {
                'success': False,
                'message': detailed_message,
                'bucket_name': self.bucket_name
            }
            
def test_step1():
    """
    1단계 테스트 함수
    
    S3 클라이언트 설정과 연결 상태만 확인합니다.
    모델 다운로드나 예측은 하지 않습니다.
    """
    print("🧪 1단계 테스트: 기본 구조와 S3 연결")
    print("=" * 50)
    
    # 1. 서비스 객체 생성
    print("📦 서비스 객체 생성 중...")
    service = MoviePredictionService()
    
    # 2. S3 연결 상태 확인
    print("\n🔍 S3 연결 상태 확인 중...")
    connection_result = service.check_s3_connection()
    
    # 3. 결과 출력
    print(f"\n📊 연결 테스트 결과:")
    print(f"   성공 여부: {connection_result['success']}")
    print(f"   메시지: {connection_result['message']}")
    print(f"   버킷명: {connection_result['bucket_name']}")
    
    # 4. 다음 단계 안내
    if connection_result['success']:
        print(f"\n🎉 1단계 성공!")
        print(f"✅ S3 클라이언트 설정: 완료")
        print(f"✅ S3 버킷 연결: 완료")
        print(f"\n🎯 1단계 완료! 이제 커밋 후 2단계 진행 가능")
        print(f"   → 다음: 2단계 - S3에서 모델 다운로드 기능 추가")
        return True
    else:
        print(f"\n❌ 1단계 실패!")
        print(f"💡 해결방법:")
        print(f"   1. AWS 인증 정보 설정 확인")
        print(f"   2. 버킷명이 정확한지 확인: {service.bucket_name}")
        print(f"   3. 인터넷 연결 상태 확인")
        return False


# 이 파일을 직접 실행할 때만 테스트 함수 실행
if __name__ == "__main__":
    test_step1()