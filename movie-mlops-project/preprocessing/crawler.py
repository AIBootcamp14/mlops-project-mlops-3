import os
import json
import time
import requests

class TMDBCrawler:
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
        params = {
            "api_key": self._api_key,
            "language": self._language,
            "region": self._region,
            "page": page
        }
        response = requests.get(f"{self._base_url}/popular", params=params)
        if not response.status_code == 200:
            print(f"❌ Error fetching page {page}: Status {response.status_code}")  # 🔥 추가: 에러 로깅
            return []  # 🔥 수정: None 대신 빈 리스트 반환
        return json.loads(response.text)["results"]
    
    def get_bulk_popular_movies(self, start_page, end_page):
        movies = []
        total_pages = end_page - start_page + 1
        
        print(f"🎬 Starting to crawl {total_pages} pages (approximately {total_pages * 20} movies)")  # 🔥 추가: 진행상황 출력
        
        for page in range(start_page, end_page + 1):
            print(f"📄 Crawling page {page}/{end_page}... ", end="")  # 🔥 추가: 페이지별 진행상황
            
            page_movies = self.get_popular_movies(page)
            if page_movies:  # 🔥 수정: None 체크 개선
                movies.extend(page_movies)
                print(f"✅ Got {len(page_movies)} movies (Total: {len(movies)})")  # 🔥 추가: 수집된 영화 수 출력
            else:
                print(f"⚠️ No movies found")  # 🔥 추가: 실패 알림
            
            # 🔥 추가: 마지막 페이지가 아닐 때만 대기
            if page < end_page:
                time.sleep(self._request_interval_seconds)
        
        print(f"🎯 Crawling completed! Total movies collected: {len(movies)}")  # 🔥 추가: 완료 메시지
        return movies
  
    @staticmethod
    def save_movies_to_json_file(movies, dst="./result", filename="popular"):
        # 🔥 추가: 디렉토리 생성
        os.makedirs(dst, exist_ok=True)
        
        data = {"movies": movies}
        file_path = os.path.join(dst, f"{filename}.json")
        
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)  # 🔥 수정: json.dump 사용, 한글 지원, 들여쓰기 추가
        
        print(f"💾 Movies saved to: {file_path}")  # 🔥 추가: 저장 완료 메시지
        print(f"📊 File size: {os.path.getsize(file_path) / 1024:.2f} KB")  # 🔥 추가: 파일 크기 정보


# 🔥 추가: 메인 실행 부분 (1000개 영화 수집)
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
    
    print("🚀 TMDB Movie Crawler Started")
    print("=" * 50)
    
    # 크롤러 초기화
    crawler = TMDBCrawler()
    
    try:
        # 🔥 핵심 수정: 1~50페이지 크롤링 (약 1000개 영화)
        movies = crawler.get_bulk_popular_movies(1, 50)
        
        if movies:
            # 🔥 수정: 파일명을 popular_1000으로 변경
            TMDBCrawler.save_movies_to_json_file(movies, filename="popular_1000")
            
            print("\n📈 Collection Summary:")
            print(f"   Total Movies: {len(movies)}")
            print(f"   Average Rating: {sum(movie.get('vote_average', 0) for movie in movies) / len(movies):.2f}")
            print(f"   Date Range: {min(movie.get('release_date', '') for movie in movies if movie.get('release_date'))} ~ {max(movie.get('release_date', '') for movie in movies if movie.get('release_date'))}")
            
        else:
            print("❌ No movies were collected!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Crawling interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
    
    print("\n✅ Crawler finished!")