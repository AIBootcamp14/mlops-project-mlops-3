import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TMDBDataPreprocessor:
    """TMDB 영화 데이터 전처리 파이프라인"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        
    def step1_load_and_inspect(self, json_file_path):
        """1단계: 데이터 로드 및 기본 탐색"""
        print("=== STEP 1: 데이터 로드 및 탐색 ===")
        
        # JSON 파일 로드
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 영화 리스트 추출
        movies = data.get('movies', data) if isinstance(data, dict) else data
        
        # DataFrame 생성
        df = pd.DataFrame(movies)
        
        print(f"원본 데이터 크기: {df.shape}")
        print(f"컬럼: {list(df.columns)}")
        print(f"\n결측값 현황:")
        print(df.isnull().sum())
        
        print(f"\n기본 통계:")
        if 'vote_average' in df.columns:
            print(f"평점 범위: {df['vote_average'].min()} ~ {df['vote_average'].max()}")
            print(f"평점 평균: {df['vote_average'].mean():.2f}")
        
        return df
    
    def step2_data_cleaning(self, df):
        """2단계: 데이터 정제"""
        print("\n=== STEP 2: 데이터 정제 ===")
        
        original_size = len(df)
        
        # 1. 중복 제거
        df = df.drop_duplicates(subset=['id'], keep='first')
        print(f"중복 제거: {original_size} → {len(df)}")
        
        # 2. 필수 컬럼 확인 및 결측값 제거
        required_columns = ['id', 'vote_average', 'vote_count', 'popularity']
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            print(f"⚠️ 필수 컬럼 누락: {missing_required}")
            return None
        
        # 3. 이상값 제거
        # 평점이 0이거나 투표수가 너무 적은 영화 제거
        before_filter = len(df)
        df = df[df['vote_average'] > 0]
        df = df[df['vote_count'] >= 10]  # 최소 10표 이상
        df = df[df['vote_average'] <= 10]  # 평점 10 이하
        print(f"이상값 제거: {before_filter} → {len(df)}")
        
        # 4. 데이터 타입 정리
        numeric_columns = ['vote_average', 'vote_count', 'popularity']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. 결측값이 있는 행 제거
        df = df.dropna(subset=required_columns)
        print(f"최종 정제된 데이터: {len(df)}개 영화")
        
        return df
    
    def step3_feature_engineering(self, df):
        """3단계: 특성 공학"""
        print("\n=== STEP 3: 특성 공학 ===")
        
        # 날짜 관련 특성
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
            df['release_quarter'] = df['release_date'].dt.quarter
            
            # 영화 나이 (2024년 기준)
            current_year = 2024
            df['movie_age'] = current_year - df['release_year']
            
            # 시즌성 특성
            df['is_summer_release'] = df['release_month'].isin([6,7,8]).astype(int)
            df['is_holiday_release'] = df['release_month'].isin([11,12]).astype(int)
            df['is_spring_release'] = df['release_month'].isin([3,4,5]).astype(int)
        
        # 장르 관련 특성
        if 'genre_ids' in df.columns:
            # genre_ids가 문자열인 경우 리스트로 변환
            def parse_genre_ids(genre_ids):
                if isinstance(genre_ids, str):
                    try:
                        return json.loads(genre_ids.replace("'", '"'))
                    except:
                        return []
                elif isinstance(genre_ids, list):
                    return genre_ids
                else:
                    return []
            
            df['genre_ids'] = df['genre_ids'].apply(parse_genre_ids)
            df['primary_genre'] = df['genre_ids'].apply(lambda x: x[0] if len(x) > 0 else 0)
            df['genre_count'] = df['genre_ids'].apply(len)
            
            # 주요 장르 원핫 인코딩
            major_genres = {
                28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
                80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
                14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
                9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
                53: 'Thriller', 10752: 'War', 37: 'Western'
            }
            
            for genre_id, genre_name in major_genres.items():
                df[f'is_{genre_name.lower().replace(" ", "_")}'] = df['genre_ids'].apply(
                    lambda x: 1 if genre_id in x else 0
                )
        
        # 언어 관련 특성
        if 'original_language' in df.columns:
            df['is_english'] = (df['original_language'] == 'en').astype(int)
            df['is_korean'] = (df['original_language'] == 'ko').astype(int)
            df['is_non_english'] = (df['original_language'] != 'en').astype(int)
        
        # 인기도 관련 특성
        df['log_popularity'] = np.log1p(df['popularity'])
        df['log_vote_count'] = np.log1p(df['vote_count'])
        
        # 투표 효율성 (인기도 대비 투표수)
        df['vote_efficiency'] = df['vote_count'] / (df['popularity'] + 1)
        
        # 평점 구간
        df['rating_tier'] = pd.cut(df['vote_average'], 
                                  bins=[0, 5, 6.5, 8, 10], 
                                  labels=['Low', 'Medium', 'High', 'Excellent'])
        
        # 인기도 구간
        df['popularity_tier'] = pd.qcut(df['popularity'], 
                                       q=5, 
                                       labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
                                       duplicates='drop')
        
        # 콘텐츠 관련 특성
        if 'adult' in df.columns:
            df['is_adult'] = df['adult'].astype(int)
        
        # 제목 관련 특성
        if 'title' in df.columns:
            df['title_length'] = df['title'].str.len()
            df['title_word_count'] = df['title'].str.split().str.len()
        
        # 줄거리 관련 특성
        if 'overview' in df.columns:
            df['overview_length'] = df['overview'].fillna('').str.len()
            df['has_overview'] = (df['overview'].notna() & (df['overview'] != '')).astype(int)
        
        # 이미지 관련 특성
        if 'poster_path' in df.columns:
            df['has_poster'] = df['poster_path'].notna().astype(int)
        if 'backdrop_path' in df.columns:
            df['has_backdrop'] = df['backdrop_path'].notna().astype(int)
        
        print(f"특성 공학 완료. 총 특성 수: {len(df.columns)}")
        
        return df
    
    def step4_handle_missing_values(self, df):
        """4단계: 결측값 처리"""
        print("\n=== STEP 4: 결측값 처리 ===")
        
        print("결측값 처리 전:")
        missing_before = df.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # 수치형 컬럼 결측값 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                if col in ['release_year', 'movie_age']:
                    # 연도 관련은 중앙값으로
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # 나머지는 평균으로
                    df[col] = df[col].fillna(df[col].mean())
        
        # 범주형 컬럼 결측값 처리
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                if col in ['rating_tier', 'popularity_tier']:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                else:
                    df[col] = df[col].fillna('Unknown')
        
        print("결측값 처리 후:")
        missing_after = df.isnull().sum()
        print(missing_after[missing_after > 0])
        
        return df
    
    def step5_feature_scaling(self, df, target_column='vote_average'):
        """5단계: 특성 스케일링"""
        print("\n=== STEP 5: 특성 스케일링 ===")
        
        # 타겟 변수 분리
        if target_column in df.columns:
            y = df[target_column].copy()
            X = df.drop(columns=[target_column])
        else:
            print(f"⚠️ 타겟 컬럼 '{target_column}'이 없습니다.")
            return df
        
        # 수치형 특성만 선택
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # ID, 바이너리 특성 제외
        exclude_features = ['id', 'content_id'] + [col for col in numeric_features if X[col].nunique() <= 2]
        numeric_features = [col for col in numeric_features if col not in exclude_features]
        
        print(f"스케일링 대상 특성: {len(numeric_features)}개")
        
        # StandardScaler 적용
        X_numeric_scaled = self.scaler.fit_transform(X[numeric_features])
        X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, 
                                          columns=[f"{col}_scaled" for col in numeric_features],
                                          index=X.index)
        
        # 범주형 특성 인코딩
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[f"{col}_encoded"] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # 최종 특성 결합
        final_features = []
        
        # 원본 수치형 특성 (바이너리, ID 포함)
        original_numeric = [col for col in X.select_dtypes(include=[np.number]).columns 
                           if col not in numeric_features]
        final_features.extend(original_numeric)
        
        # 스케일링된 특성
        final_features.extend([f"{col}_scaled" for col in numeric_features])
        
        # 인코딩된 범주형 특성
        final_features.extend([f"{col}_encoded" for col in categorical_features])
        
        # 최종 데이터셋 구성
        X_final = pd.concat([
            X[original_numeric],
            X_numeric_scaled_df,
            X[[f"{col}_encoded" for col in categorical_features]]
        ], axis=1)
        
        # 타겟 변수 추가
        final_df = pd.concat([X_final, y], axis=1)
        
        print(f"최종 특성 수: {len(X_final.columns)}")
        
        return final_df
    
    def step6_train_test_split(self, df, target_column='vote_average', test_size=0.2):
        """6단계: 학습/테스트 데이터 분할"""
        print("\n=== STEP 6: 데이터 분할 ===")
        
        # 특성과 타겟 분리
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 시간순 분할 (release_year가 있는 경우)
        if 'release_year' in df.columns:
            # 최신 20%의 영화를 테스트셋으로
            year_threshold = df['release_year'].quantile(0.8)
            
            train_mask = df['release_year'] <= year_threshold
            test_mask = df['release_year'] > year_threshold
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            print(f"시간순 분할 적용 (기준연도: {year_threshold})")
        else:
            # 랜덤 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
            )
            print("랜덤 분할 적용")
        
        print(f"학습 데이터: {len(X_train)}개")
        print(f"테스트 데이터: {len(X_test)}개")
        print(f"학습 데이터 평점 분포: {y_train.describe()}")
        print(f"테스트 데이터 평점 분포: {y_test.describe()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_preprocessing_summary(self, df):
        """전처리 결과 요약"""
        print("\n=== 전처리 완료 요약 ===")
        print(f"최종 데이터 크기: {df.shape}")
        print(f"특성 수: {len(df.columns) - 1}")  # 타겟 제외
        print(f"평점 분포:")
        print(df['vote_average'].describe())
        
        # 특성별 타입 확인
        print(f"\n특성 타입별 개수:")
        print(df.dtypes.value_counts())
        
        return df
    
    def run_full_pipeline(self, json_file_path, target_column='vote_average'):
        """전체 전처리 파이프라인 실행"""
        print("🎬 TMDB 영화 평점 예측 데이터 전처리 시작")
        print("=" * 50)
        
        # 전체 파이프라인 실행
        df = self.step1_load_and_inspect(json_file_path)
        if df is None:
            return None
            
        df = self.step2_data_cleaning(df)
        if df is None:
            return None
            
        df = self.step3_feature_engineering(df)
        df = self.step4_handle_missing_values(df)
        df = self.step5_feature_scaling(df, target_column)
        
        # 최종 요약
        df = self.get_preprocessing_summary(df)
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = self.step6_train_test_split(df, target_column)
        
        print("\n✅ 전처리 완료!")
        
        return {
            'processed_df': df,
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }
    
    def save_processed_data(self, results, output_dir='./result'):
        """전처리된 데이터 저장"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 전체 데이터셋 저장
        results['processed_df'].to_csv(f'{output_dir}/tmdb_processed_full.csv', index=False)
        
        # 학습/테스트 데이터 저장
        pd.concat([results['X_train'], results['y_train']], axis=1).to_csv(
            f'{output_dir}/tmdb_train.csv', index=False
        )
        pd.concat([results['X_test'], results['y_test']], axis=1).to_csv(
            f'{output_dir}/tmdb_test.csv', index=False
        )
        
        # 특성 목록 저장
        with open(f'{output_dir}/feature_names.json', 'w') as f:
            json.dump(results['feature_names'], f, indent=2)
        
        print(f"💾 전처리된 데이터가 {output_dir}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 전처리기 초기화
    preprocessor = TMDBDataPreprocessor()
    
    # 전체 파이프라인 실행
    results = preprocessor.run_full_pipeline('./result/popular.json')
    
    if results:
        # 결과 저장 (자동)
        preprocessor.save_processed_data(results)
        
        # 추가 수동 저장 (확실히 하기 위해)
        print("\n💾 CSV 파일 저장 확인:")
        
        # 개별 DataFrame 저장
        results['processed_df'].to_csv('./result/full_dataset.csv', index=False, encoding='utf-8')
        print("✅ 전체 데이터셋: ./result/full_dataset.csv")
        
        results['X_train'].to_csv('./result/X_train.csv', index=False, encoding='utf-8')
        results['y_train'].to_csv('./result/y_train.csv', index=False, encoding='utf-8')
        print("✅ 학습 데이터: ./result/X_train.csv, ./result/y_train.csv")
        
        results['X_test'].to_csv('./result/X_test.csv', index=False, encoding='utf-8')
        results['y_test'].to_csv('./result/y_test.csv', index=False, encoding='utf-8')
        print("✅ 테스트 데이터: ./result/X_test.csv, ./result/y_test.csv")
        
        print(f"\n📊 저장된 데이터 정보:")
        print(f"- 전체 데이터 크기: {results['processed_df'].shape}")
        print(f"- 학습 데이터 크기: {results['X_train'].shape}")
        print(f"- 테스트 데이터 크기: {results['X_test'].shape}")
        print(f"- 특성 개수: {len(results['feature_names'])}")
        
        # 저장된 파일 크기 확인
        import os
        if os.path.exists('./result/full_dataset.csv'):
            size = os.path.getsize('./result/full_dataset.csv') / 1024 / 1024
            print(f"- 파일 크기: {size:.2f} MB")