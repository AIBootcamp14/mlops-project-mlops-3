import os
import json
import time
import requests
from datetime import datetime

class TestDataCrawler:
    def __init__(
        self, region="KR", 
        language="ko-KR", 
        image_language="ko", 
        request_interval_seconds=0.4
    ):
        self._base_url = os.environ.get("TMDB_BASE_URL")
        self._api_key = os.environ.get("TMDB_API_KEY")
        self._region = region
        self._language = language
        self._image_language = image_language
        self._request_interval_seconds = request_interval_seconds
    
    def get_popular_movies(self, page):
        """인기 영화 목록 가져오기"""
        params = {
            "api_key": self._api_key,
            "language": self._language,
            "region": self._region,
            "page": page
        }
        response = requests.get(f"{self._base_url}/popular", params=params)
        if not response.status_code == 200:
            print(f"❌ Error fetching popular page {page}: Status {response.status_code}")
            return []
        return json.loads(response.text)["results"]
    
    def get_upcoming_movies(self, page):
        """개봉 예정 영화 목록 가져오기"""
        params = {
            "api_key": self._api_key,
            "language": self._language,
            "region": self._region,
            "page": page
        }
        response = requests.get(f"{self._base_url}/upcoming", params=params)
        if not response.status_code == 200:
            print(f"❌ Error fetching upcoming page {page}: Status {response.status_code}")
            return []
        return json.loads(response.text)["results"]
    
    def collect_test_popular_extended(self):
        """테스트 데이터 1: 인기 영화 21-30페이지 (200편)"""
        print("\n📊 Collecting Test Data 1: Popular Extended (pages 21-30)")
        print("=" * 50)
        
        movies = []
        for page in range(21, 31):  # 21-30 페이지
            print(f"📄 Crawling popular page {page}/30... ", end="")
            
            page_movies = self.get_popular_movies(page)
            if page_movies:
                movies.extend(page_movies)
                print(f"✅ Got {len(page_movies)} movies (Total: {len(movies)})")
            else:
                print(f"⚠️ No movies found")
            
            if page < 30:
                time.sleep(self._request_interval_seconds)
        
        print(f"✨ Collected {len(movies)} popular extended movies")
        return movies
    
    def collect_test_upcoming_by_rating(self):
        """테스트 데이터 2, 3: 개봉 예정 영화 평점 유무로 분류"""
        print("\n📊 Collecting Test Data 2 & 3: Upcoming Movies (pages 1-5)")
        print("=" * 50)
        
        movies_with_ratings = []
        movies_without_ratings = []
        
        for page in range(1, 6):  # 1-5 페이지
            print(f"📄 Crawling upcoming page {page}/5... ", end="")
            
            page_movies = self.get_upcoming_movies(page)
            if page_movies:
                # 평점 유무로 분류
                for movie in page_movies:
                    if movie.get('vote_count', 0) > 0 and movie.get('vote_average', 0) > 0:
                        movies_with_ratings.append(movie)
                    else:
                        movies_without_ratings.append(movie)
                
                print(f"✅ Got {len(page_movies)} movies (Rated: {len(movies_with_ratings)}, Unrated: {len(movies_without_ratings)})")
            else:
                print(f"⚠️ No movies found")
            
            if page < 5:
                time.sleep(self._request_interval_seconds)
        
        print(f"✨ Collected {len(movies_with_ratings)} rated upcoming movies")
        print(f"✨ Collected {len(movies_without_ratings)} unrated upcoming movies")
        
        # 목표 수량에 맞게 조정
        movies_with_ratings = movies_with_ratings[:100]  # 최대 100편
        movies_without_ratings = movies_without_ratings[:50]  # 최대 50편
        
        return movies_with_ratings, movies_without_ratings
    
    @staticmethod
    def save_movies_to_json_file(movies, dst="./result", filename="test_data"):
        """영화 데이터를 JSON 파일로 저장"""
        os.makedirs(dst, exist_ok=True)
        
        # 메타데이터 추가
        data = {
            "movies": movies,
            "count": len(movies),
            "crawled_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_type": filename
        }
        
        file_path = os.path.join(dst, f"{filename}.json")
        
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Movies saved to: {file_path}")
        print(f"📊 File size: {os.path.getsize(file_path) / 1024:.2f} KB")
        
        # 간단한 통계 출력
        if movies:
            avg_rating = sum(movie.get('vote_average', 0) for movie in movies) / len(movies)
            print(f"📈 Average rating: {avg_rating:.2f}")


if __name__ == "__main__":
    # 환경변수 확인
    if not os.environ.get("TMDB_API_KEY"):
        print("❌ TMDB_API_KEY 환경변수가 설정되지 않았습니다!")
        print("다음 명령어로 설정하세요:")
        print("export TMDB_API_KEY='your_api_key_here'")
        print("export TMDB_BASE_URL='https://api.themoviedb.org/3/movie'")
        exit(1)
    
    if not os.environ.get("TMDB_BASE_URL"):
        print("❌ TMDB_BASE_URL 환경변수가 설정되지 않았습니다!")
        print("export TMDB_BASE_URL='https://api.themoviedb.org/3/movie'")
        exit(1)
    
    print("🚀 TMDB Test Data Crawler Started")
    print("=" * 50)
    
    # 크롤러 초기화
    crawler = TestDataCrawler()
    
    try:
        # 1. 인기 영화 21-30페이지 수집
        popular_extended = crawler.collect_test_popular_extended()
        if popular_extended:
            TestDataCrawler.save_movies_to_json_file(
                popular_extended, 
                filename="test_popular_extended"
            )
        
        print("\n" + "=" * 50)
        
        # 2, 3. 개봉 예정 영화 평점 유무로 분류하여 수집
        upcoming_rated, upcoming_unrated = crawler.collect_test_upcoming_by_rating()
        
        if upcoming_rated:
            TestDataCrawler.save_movies_to_json_file(
                upcoming_rated, 
                filename="test_upcoming_rated"
            )
        
        if upcoming_unrated:
            TestDataCrawler.save_movies_to_json_file(
                upcoming_unrated, 
                filename="test_upcoming_unrated"
            )
        
        # 전체 요약
        print("\n" + "=" * 50)
        print("📊 Test Data Collection Summary:")
        print(f"   1. Popular Extended: {len(popular_extended)} movies")
        print(f"   2. Upcoming Rated: {len(upcoming_rated)} movies")
        print(f"   3. Upcoming Unrated: {len(upcoming_unrated)} movies")
        print(f"   Total Test Data: {len(popular_extended) + len(upcoming_rated) + len(upcoming_unrated)} movies")
        
    except KeyboardInterrupt:
        print("\n⏹️ Crawling interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test Data Crawler finished!")