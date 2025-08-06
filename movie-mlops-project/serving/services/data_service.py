import os          # 파일 시스템 작업 (파일 존재 확인, 경로 처리)
import logging     # 로그 출력 (정보, 경고, 에러 메시지)
import pandas as pd # CSV 파일 읽기 및 데이터 처리
import numpy as np  # 수치 계산 및 배열 처리

# 로깅 설정 - 이 모듈의 로거 생성
logger = logging.getLogger(__name__)

class DataService:
    """
    CSV 데이터 로드 및 전처리 서비스 클래스
    
    1. tmdb_test.csv 파일을 pandas DataFrame으로 로드
    2. 데이터가 예상한 형태인지 검증 (컬럼명, 데이터 타입 등)
    3. 모델이 사용할 수 있는 형태로 데이터 변환
    4. 데이터에 문제가 있으면 에러 처리
    """
    def __init__(self, csv_path=None):
        """
        DataService 객체 초기화 함수

        - 객체가 생성될 때 자동으로 실행되는 함수
        - 필요한 설정값들을 저장
        
        Args:
            csv_path (str, optional): CSV 파일 경로
                                    None이면 기본 경로 사용
        """
        
        # === 1. CSV 파일 경로 설정 ===
        if csv_path is None:
            # 현재 파일(data_service.py)의 위치를 기준으로 계산
            current_dir = os.path.dirname(os.path.abspath(__file__))  # services/
            serving_dir = os.path.dirname(current_dir)                # serving/
            project_dir = os.path.dirname(serving_dir)                # movie-mlops-project/
        
            # CSV 파일 절대 경로 계산
            self.csv_path = os.path.join(project_dir, "preprocessing", "result", "tmdb_test.csv")
        else:
            self.csv_path = csv_path
        
        # === 2. 데이터 상태를 추적하는 변수들 ===
        self.data = None                    # 로드된 pandas DataFrame 저장소
        self.is_data_loaded = False         # 데이터 로드 완료 여부 (True/False)
        self.data_shape = None              # 데이터 크기 (행 수, 열 수)
        self.data_columns = None            # 컬럼 이름 목록
        self.data_info = {}                 # 데이터 메타정보 (통계, 타입 등)
        
        # === 3. 예상 데이터 스키마 (검증용) ===
        # - 어떤 컬럼들이 있어야 하는지 미리 정의
        # - 실제 데이터와 비교해서 일치하는지 확인
        self.expected_columns = [
            # 영화 기본 정보
            'id',                           # 영화 고유 ID
            
            # 출시 시기 관련 (원-핫 인코딩)
            'is_summer_release',            # 여름 출시 여부 (0 또는 1)
            'is_holiday_release',           # 휴일 출시 여부 (0 또는 1)  
            'is_spring_release',            # 봄 출시 여부 (0 또는 1)
            
            # 장르 관련 (원-핫 인코딩)
            'is_action', 'is_adventure', 'is_animation', 'is_comedy',
            'is_crime', 'is_documentary', 'is_drama', 'is_family',
            'is_fantasy', 'is_history', 'is_horror', 'is_music',
            'is_mystery', 'is_romance', 'is_science_fiction',
            'is_thriller', 'is_war', 'is_western',
            
            # 언어 관련
            'is_english', 'is_korean', 'is_non_english',
            
            # 영화 속성
            'is_adult',                     # 성인 영화 여부
            'has_overview', 'has_poster', 'has_backdrop',  # 컨텐츠 존재 여부
            
            # 수치형 특성들 (스케일링된 값)
            'popularity_scaled',            # 인기도 (정규화된 값)
            'vote_count_scaled',            # 투표 수 (정규화된 값)
            'release_year_scaled',          # 출시 연도 (정규화된 값)
            'release_month_scaled',         # 출시 월 (정규화된 값)
            'release_quarter_scaled',       # 출시 분기 (정규화된 값)
            'movie_age_scaled',             # 영화 나이 (정규화된 값)
            'primary_genre_scaled',         # 주 장르 (정규화된 값)
            'genre_count_scaled',           # 장르 개수 (정규화된 값)
            'log_popularity_scaled',        # 로그 인기도 (정규화된 값)
            'log_vote_count_scaled',        # 로그 투표수 (정규화된 값)
            'vote_efficiency_scaled',       # 투표 효율성 (정규화된 값)
            'title_length_scaled',          # 제목 길이 (정규화된 값)
            'title_word_count_scaled',      # 제목 단어 수 (정규화된 값)
            'overview_length_scaled',       # 개요 길이 (정규화된 값)
            
            # 인코딩된 특성들
            'backdrop_path_encoded',        # 배경 이미지 경로 (인코딩된 값)
            'genre_ids_encoded',            # 장르 ID (인코딩된 값)
            'original_language_encoded',    # 원본 언어 (인코딩된 값)
            'original_title_encoded',       # 원본 제목 (인코딩된 값)
            'overview_encoded',             # 개요 (인코딩된 값)
            'poster_path_encoded',          # 포스터 경로 (인코딩된 값)
            'title_encoded',                # 제목 (인코딩된 값)
            'rating_tier_encoded',          # 평점 구간 (인코딩된 값)
            'popularity_tier_encoded',      # 인기도 구간 (인코딩된 값)
            
            # 타겟 변수 (예측할 값)
            'vote_average'                   # 실제 평점 (예측 대상)
        ]
        
        # === 4. 초기화 완료 로그 ===
        logger.info("✅ DataService 초기화 완료")
        logger.info(f"   📁 CSV 경로: {self.csv_path}")
        logger.info(f"   🏷️ 예상 컬럼 수: {len(self.expected_columns)}개")
        
        
    def check_file_exists(self):
        """
        CSV 파일 존재 여부 확인 함수
        
        Returns:
            dict: 파일 존재 여부와 관련 정보
                {
                    'exists': True/False,        # 파일 존재 여부
                    'message': '상태 메시지',      # 사용자에게 보여줄 메시지
                    'file_path': '절대경로',       # 실제 파일 경로
                    'file_size': 바이트수          # 파일 크기
                }
        """
        try:
            # === 1. 로그 출력 (디버깅용) ===
            logger.info(f"🔍 CSV 파일 존재 확인: {self.csv_path}")
            
            # === 2. 상대 경로를 절대 경로로 변환 ===
            abs_path = os.path.abspath(self.csv_path)
            
            # === 3. 파일 존재 여부 확인 ===
            if os.path.exists(abs_path):
                # 파일이 존재하는 경우
                
                # 파일 크기 확인 (바이트 단위)
                file_size = os.path.getsize(abs_path)
                
                # 성공 로그 출력
                logger.info("✅ CSV 파일 존재 확인!")
                logger.info(f"   📁 절대 경로: {abs_path}")
                logger.info(f"   📦 파일 크기: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                
                # 성공 결과 반환
                return {
                    'exists': True,
                    'message': 'CSV 파일이 존재합니다.',
                    'file_path': abs_path,
                    'file_size': file_size
                }
            else:
                # 파일이 존재하지 않는 경우
                logger.warning(f"⚠️ CSV 파일을 찾을 수 없음: {abs_path}")
                
                # 실패 결과 반환
                return {
                    'exists': False,
                    'message': f'CSV 파일을 찾을 수 없습니다: {abs_path}',
                    'file_path': abs_path,
                    'file_size': None
                }
                
        except Exception as e:
            # === 4. 예외 처리 (예상하지 못한 에러) ===
            # - 권한 없음 (Permission denied)
            # - 디스크 오류
            # - 네트워크 드라이브 연결 문제 등
            
            logger.error(f"❌ 파일 확인 중 오류: {e}")
            
            return {
                'exists': False,
                'message': f'파일 확인 중 오류 발생: {str(e)}',
                'file_path': self.csv_path,
                'file_size': None
            }
       
            
    def load_data(self):
        """
        CSV 데이터를 pandas DataFrame으로 로드
        
        Returns:
            dict: 로드 결과 {'success': bool, 'message': str, 'data_loaded': bool, ...}
        """
        # 파일 존재 확인
        file_check = self.check_file_exists()
        if not file_check['exists']:
            return {
                'success': False,
                'message': f"파일 없음: {file_check['message']}",
                'data_loaded': False,
                'shape': None
            }
        
        try:
            # CSV 파일 로드
            logger.info(f"📊 CSV 데이터 로딩 시작...")
            logger.info(f"   📁 파일: {file_check['file_path']}")
            logger.info(f"   📦 크기: {file_check['file_size']:,} bytes")
            
            # pandas로 CSV 읽기 (핵심!)
            self.data = pd.read_csv(file_check['file_path'])
            
            # 기본 정보 수집
            self.data_shape = self.data.shape                    # (행수, 열수)
            self.data_columns = list(self.data.columns)          # 컬럼명 목록
            self.is_data_loaded = True                           # 로드 완료 플래그
            
            # 데이터 품질 확인
            null_counts = self.data.isnull().sum().sum()         # 전체 결측치 개수
            duplicate_counts = self.data.duplicated().sum()      # 중복 행 개수
            
            # 메타정보 저장
            self.data_info = {
                'shape': self.data_shape,
                'columns': self.data_columns,
                'null_count': null_counts,
                'duplicate_count': duplicate_counts,
                'dtypes': self.data.dtypes.to_dict()             # 각 컬럼의 데이터 타입
            }
            
            # 성공 로그
            logger.info("✅ CSV 데이터 로드 완료!")
            logger.info(f"   📊 데이터 크기: {self.data_shape[0]:,} rows × {self.data_shape[1]} columns")
            logger.info(f"   🏷️ 컬럼 수: {len(self.data_columns)}")
            logger.info(f"   🔍 결측치: {null_counts:,}개")
            logger.info(f"   🔍 중복 행: {duplicate_counts:,}개")
            
            return {
                'success': True,
                'message': 'CSV 데이터 로드가 성공적으로 완료되었습니다.',
                'data_loaded': True,
                'shape': self.data_shape,
                'columns_count': len(self.data_columns),
                'null_count': null_counts,
                'duplicate_count': duplicate_counts
            }
            
        except Exception as e:
            # 실패 처리
            logger.error(f"❌ CSV 데이터 로드 실패: {e}")
            
            # 상태 초기화
            self.data = None
            self.is_data_loaded = False
            self.data_shape = None
            self.data_columns = None
            
            return {
                'success': False,
                'message': f'CSV 데이터 로드 실패: {str(e)}',
                'data_loaded': False,
                'shape': None
            }
            
            
# =============================================================================
# 간단한 테스트 함수 (동작 확인용)
# =============================================================================
def test_basic_functionality():
    """
    기본 기능 테스트 함수
    DataService 객체 생성과 파일 존재 확인만 테스트
    """
    print("🧪 DataService 기본 기능 테스트")
    print("=" * 40)
    
    # 1. 객체 생성
    print("📦 DataService 객체 생성 중...")
    service = DataService()
    print("✅ 객체 생성 완료")
    
    # 2. 설정 확인
    print(f"📁 CSV 경로: {service.csv_path}")
    print(f"🏷️ 예상 컬럼 수: {len(service.expected_columns)}개")
    
    # 3. 파일 존재 확인
    print("\n🔍 파일 존재 확인 중...")
    result = service.check_file_exists()
    
    print(f"📊 결과:")
    print(f"   존재 여부: {result['exists']}")
    print(f"   메시지: {result['message']}")
    if result['file_size']:
        print(f"   파일 크기: {result['file_size']:,} bytes")
    
    if result['exists']:
        print("🎉 기본 테스트 성공!")
        return True  # ✅ 수정: return True 추가
    else:
        print("⚠️ CSV 파일을 확인해주세요.")
        return False # ✅ 수정: return False 추가

def test_data_loading():
    """🧪 2차 테스트: CSV 데이터 로딩"""
    print("🧪 2차 테스트: CSV 데이터 로딩")
    print("=" * 40)
    
    # 1. 서비스 생성 및 데이터 로드
    service = DataService()
    result = service.load_data()
    
    # 2. 결과 출력
    print(f"📊 로딩 결과: {result['success']}")
    print(f"메시지: {result['message']}")
    
    if result['success']:
        print(f"📏 데이터 크기: {result['shape'][0]:,} × {result['shape'][1]}")
        print(f"🏷️ 컬럼 수: {result['columns_count']}")
        print(f"🕳️ 결측치: {result['null_count']}")
        print(f"🔄 중복 행: {result['duplicate_count']}")
        
        # 3. 샘플 데이터 미리보기
        print(f"\n👀 첫 5개 컬럼: {service.data_columns[:5]}")
        print(f"📊 데이터 미리보기:")
        print(service.data.iloc[:3, :5].to_string())
        
        print(f"\n🎉 2차 테스트 성공!")
        return True
    else:
        print(f"❌ 2차 테스트 실패!")
        return False

def test_complete():
    """🧪 전체 테스트: 1차 + 2차"""
    print("🚀 DataService 전체 테스트")
    print("=" * 50)
    
    # 1차 테스트
    print("1️⃣ 파일 존재 확인...")
    step1 = test_basic_functionality()
    
    if step1:  # 이제 step1이 제대로 True/False 값을 가집니다
        print("\n2️⃣ 데이터 로딩...")
        step2 = test_data_loading()
        
        if step2:
            print(f"\n🏆 전체 테스트 성공!")
            print(f"✅ 파일 확인: 완료")
            print(f"✅ 데이터 로딩: 완료")
            print(f"🎯 2단계 완료! 다음: 3단계 - 예측 서비스")
            return True
    
    print(f"⚠️ 테스트 실패")
    return False

# ✅ 수정: debug_file_paths() 함수 추가 (간단 버전)
def debug_file_paths():
    """🔍 파일 경로 디버깅 함수 (간단 버전)"""
    print("🔍 파일 경로 디버깅")
    print("=" * 50)
    
    current_dir = os.getcwd()
    print(f"📍 현재 디렉토리: {current_dir}")
    
    # 현재 설정된 경로 확인
    service = DataService()
    print(f"📁 설정된 CSV 경로: {service.csv_path}")
    
    abs_path = os.path.abspath(service.csv_path)
    exists = os.path.exists(abs_path)
    
    print(f"📄 절대 경로: {abs_path}")
    print(f"📊 파일 존재: {'✅ 예' if exists else '❌ 아니오'}")
    
    if exists:
        size = os.path.getsize(abs_path)
        print(f"📦 파일 크기: {size:,} bytes")

# 실행 부분 업데이트
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            test_basic_functionality()
        elif sys.argv[1] == "2": 
            test_data_loading()
        elif sys.argv[1] == "test":
            test_complete()
        elif sys.argv[1] == "debug":
            debug_file_paths()  # 이제 함수가 정의되어 있음
        else:
            print("사용법:")
            print("  python data_service.py        # 2차 테스트")
            print("  python data_service.py 1      # 1차 테스트")
            print("  python data_service.py 2      # 2차 테스트")
            print("  python data_service.py test   # 전체 테스트")
            print("  python data_service.py debug  # 경로 디버깅")
    else:
        test_data_loading()