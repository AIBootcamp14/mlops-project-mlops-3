# TMDB 영화 데이터 전처리 파이프라인

## 개요
TMDB(The Movie Database) API를 통해 수집한 영화 데이터를 머신러닝 모델 학습에 적합한 형태로 전처리하는 파이프라인입니다.

## 주요 기능

### 1. 데이터 수집 (Crawling)
- **인기 영화**: 1-50페이지 (약 1,000편)
- **테스트 데이터**:
  - `test_popular_extended`: 인기 영화 21-30페이지 (200편)
  - `test_upcoming_rated`: 평점 있는 개봉 예정 영화 (최대 100편)
  - `test_upcoming_unrated`: 평점 없는 개봉 예정 영화 (최대 50편)

### 2. 전처리 파이프라인
1. **데이터 로드 및 검사**
2. **데이터 정제**
   - 중복 제거
   - 이상값 제거
   - 필수 컬럼 확인
3. **특성 공학**
   - 날짜 관련 특성 (연도, 월, 분기, 영화 나이)
   - 장르 관련 특성 (원핫 인코딩, 주요 장르, 장르 수)
   - 언어 관련 특성 (영어/한국어/기타)
   - 인기도 관련 특성 (로그 변환, 투표 효율성)
   - 콘텐츠 관련 특성 (제목/줄거리 길이, 이미지 유무)
4. **결측값 처리**
5. **특성 스케일링**
   - StandardScaler (수치형 특성)
   - LabelEncoder (범주형 특성)
6. **데이터 분할**
   - 시간순 분할 (release_year 기준)
   - 테스트 데이터는 분할하지 않음

## 사용 방법

### 환경 설정
```bash
# 환경 변수 설정
export TMDB_API_KEY='your_api_key_here'
export TMDB_BASE_URL='https://api.themoviedb.org/3/movie'

# 필요한 패키지 설치
pip install pandas numpy scikit-learn requests python-dotenv
```

### 실행
```bash
# 전체 파이프라인 실행 (크롤링 + 전처리)
python main.py
```

### 개별 실행
```python
# 크롤러만 실행
from crawler import TMDBCrawler
crawler = TMDBCrawler()
movies = crawler.get_bulk_popular_movies(1, 50)
crawler.save_movies_to_json_file(movies)

# 전처리만 실행
from preprocessing import TMDBDataPreprocessor
preprocessor = TMDBDataPreprocessor()
results = preprocessor.run_full_pipeline('./result/popular.json')
preprocessor.save_processed_data(results)
```

## 출력 파일 구조

### result/ 폴더
```
result/
├── 크롤링 데이터 (JSON)
│   ├── popular.json                        # 인기 영화 1,000편
│   ├── test_popular_extended.json          # 테스트용 인기 영화 200편
│   ├── test_upcoming_rated.json            # 평점 있는 개봉예정 영화
│   └── test_upcoming_unrated.json          # 평점 없는 개봉예정 영화
│
├── 전처리 데이터 (CSV)
│   ├── tmdb_processed_full.csv             # 전체 데이터
│   ├── tmdb_train.csv                      # 학습 데이터
│   ├── tmdb_test.csv                       # 테스트 데이터
│   ├── test_popular_extended_tmdb_processed_full.csv
│   ├── test_upcoming_rated_tmdb_processed_full.csv
│   └── test_upcoming_unrated_tmdb_processed_full.csv
│
└── 메타데이터
    └── feature_names.json                  # 특성 이름 목록
```

## 특성 목록
전처리 후 생성되는 주요 특성:
- **기본 정보**: id, vote_average (타겟)
- **시간 관련**: release_year, release_month, movie_age
- **장르 관련**: is_action, is_comedy, genre_count 등
- **언어 관련**: is_english, is_korean
- **인기도 관련**: popularity_scaled, vote_efficiency_scaled
- **콘텐츠 관련**: title_length, has_overview, has_poster

## 주의 사항
1. **API 제한**: TMDB API는 요청 제한이 있으므로 0.4초 간격으로 요청
2. **평점 0 처리**: 평점이 없는 영화(vote_average=0)도 처리 가능하도록 설정
3. **메모리 사용**: 대량 데이터 처리 시 메모리 사용량 주의

## 파일 설명
- `main.py`: 전체 파이프라인 실행
- `crawler.py`: TMDB API 크롤링
- `test_data_crawler.py`: 테스트 데이터 수집
- `preprocessing.py`: 데이터 전처리
- `.env`: API 키 설정 (생성 필요)

## 문의사항
문제 발생 시 이슈를 생성하거나 README를 참고하세요.