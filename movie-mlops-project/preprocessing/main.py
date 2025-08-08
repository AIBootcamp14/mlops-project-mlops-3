import pandas as pd
from dotenv import load_dotenv

from preprocessing import TMDBDataPreprocessor
from crawler import TMDBCrawler
from test_data_crawler import TestDataCrawler

load_dotenv()


def run_popular_movie_crawler():
    tmdb_crawler = TMDBCrawler()
    result = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=50)
    tmdb_crawler.save_movies_to_json_file(result, "./result", "popular")

    tmdb_preprocessor = TMDBDataPreprocessor()
    preprocessed_result = tmdb_preprocessor.run_full_pipeline("./result/popular.json")
    if preprocessed_result:
        tmdb_preprocessor.save_processed_data(preprocessed_result)


def run_test_data_crawler():
    print("\n🚀 Starting Test Data Collection...")
    print("=" * 50)
    
    test_crawler = TestDataCrawler()
    tmdb_preprocessor = TMDBDataPreprocessor()
    
    # 1. 인기 영화 21-30페이지 수집
    popular_extended = test_crawler.collect_test_popular_extended()
    if popular_extended:
        TestDataCrawler.save_movies_to_json_file(
            popular_extended, 
            filename="test_popular_extended"
        )
        # 전처리 실행 (분할 없이)
        print("\n📊 Preprocessing test_popular_extended.json...")
        preprocessed_result = tmdb_preprocessor.run_full_pipeline("./result/test_popular_extended.json", split_data=False)
        if preprocessed_result:
            tmdb_preprocessor.save_processed_data(preprocessed_result, filename_prefix="test_popular_extended")
    
    # 2, 3. 개봉 예정 영화 평점 유무로 분류하여 수집
    upcoming_rated, upcoming_unrated = test_crawler.collect_test_upcoming_by_rating()
    
    if upcoming_rated:
        TestDataCrawler.save_movies_to_json_file(
            upcoming_rated, 
            filename="test_upcoming_rated"
        )
        # 전처리 실행 (분할 없이)
        print("\n📊 Preprocessing test_upcoming_rated.json...")
        preprocessed_result = tmdb_preprocessor.run_full_pipeline("./result/test_upcoming_rated.json", split_data=False)
        if preprocessed_result:
            tmdb_preprocessor.save_processed_data(preprocessed_result, filename_prefix="test_upcoming_rated")
    
    if upcoming_unrated:
        TestDataCrawler.save_movies_to_json_file(
            upcoming_unrated, 
            filename="test_upcoming_unrated"
        )
        # 전처리 실행 (분할 없이)
        print("\n📊 Preprocessing test_upcoming_unrated.json...")
        preprocessed_result = tmdb_preprocessor.run_full_pipeline("./result/test_upcoming_unrated.json", split_data=False)
        if preprocessed_result:
            tmdb_preprocessor.save_processed_data(preprocessed_result, filename_prefix="test_upcoming_unrated")
    
    print("\n✅ Test data collection and preprocessing completed!")


if __name__ == '__main__':
    # 1. 기존 인기 영화 크롤링 (1-50페이지)
    print("📌 Step 1: Crawling popular movies (pages 1-50)...")
    run_popular_movie_crawler()
    
    # 2. 테스트 데이터 크롤링 (3종류)
    print("\n📌 Step 2: Crawling test datasets...")
    run_test_data_crawler()
    
    print("\n✅ All crawling tasks completed!")