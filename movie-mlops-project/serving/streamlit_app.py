import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# 환경변수 로드
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="🎬 영화 평점 예측 대시보드",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 최소한의 스타일링만 적용
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-bad { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# API 설정
API_BASE_URL = "http://localhost:8000"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w200"

@st.cache_data(ttl=300)  # 5분 캐시
def get_api_data(endpoint, params=None):
    """API에서 데이터 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json(), True
        else:
            return {"error": f"HTTP {response.status_code}"}, False
    except requests.exceptions.ConnectionError:
        return {"error": "API 서버에 연결할 수 없습니다"}, False
    except requests.exceptions.Timeout:
        return {"error": "요청 시간 초과"}, False
    except Exception as e:
        return {"error": str(e)}, False

@st.cache_data(ttl=3600)  # 1시간 캐시
def get_movie_details(movie_id):
    """TMDB API에서 영화 상세 정보 가져오기"""
    if not TMDB_API_KEY:
        return None
        
    try:
        base_url = TMDB_BASE_URL
        if base_url.endswith('/movie'):
            url = f"{base_url}/{movie_id}"
        else:
            url = f"{base_url}/movie/{movie_id}"
            
        params = {"api_key": TMDB_API_KEY, "language": "ko-KR"}
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'title': data.get('title', f'영화 ID {movie_id}'),
                'original_title': data.get('original_title', ''),
                'poster_path': data.get('poster_path'),
                'release_date': data.get('release_date', ''),
                'overview': data.get('overview', ''),
                'genres': [genre['name'] for genre in data.get('genres', [])],
                'runtime': data.get('runtime'),
                'vote_average': data.get('vote_average')
            }
        else:
            return None
    except Exception as e:
        return None

def get_poster_url(poster_path):
    """포스터 이미지 URL 생성"""
    if poster_path:
        return f"{TMDB_IMAGE_URL}{poster_path}"
    return None

def display_simple_movie_card(movie, movie_details=None):
    """간단하고 깔끔한 영화 카드 표시"""
    # 기본 정보 추출
    rank = movie.get('rank', 0)
    movie_id = movie.get('movie_id', 0)
    predicted_rating = movie.get('predicted_rating', 0.0)
    
    # TMDB 정보 처리
    if movie_details:
        title = movie_details.get('title', f'영화 ID {movie_id}')
        original_title = movie_details.get('original_title', '')
        release_date = movie_details.get('release_date', '')
        genres = movie_details.get('genres', [])
        poster_url = get_poster_url(movie_details.get('poster_path'))
    else:
        title = f'영화 ID {movie_id}'
        original_title = ''
        release_date = ''
        genres = []
        poster_url = None

    # 스트림릿 expander 사용 (깔끔한 카드 효과)
    with st.expander(f"🏅 {rank}위 | ⭐ {predicted_rating:.2f}", expanded=True):
        # 메인 정보와 포스터를 나란히
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 영화 제목
            display_title = title[:30] + "..." if len(title) > 30 else title
            st.subheader(display_title)
            
            # 원제목 (다른 경우에만)
            if original_title and original_title != title:
                st.caption(f"*{original_title[:35]}*")
            
            # 기본 정보
            st.write(f"**영화 ID:** {movie_id}")
            st.write(f"**🎯 예측평점:** {predicted_rating:.2f}")
            
            # 추가 정보 (있는 경우)
            if release_date:
                release_year = release_date[:4] if len(release_date) >= 4 else release_date
                st.write(f"**개봉:** {release_year}")
            
            if genres:
                genres_str = ', '.join(genres[:3])  # 최대 3개 장르
                st.write(f"**장르:** {genres_str}")
        
        with col2:
            # 포스터 이미지 (130px로 복원)
            if poster_url:
                try:
                    st.image(poster_url, width=130)
                except:
                    st.write("🎬 포스터 로드 실패")
            else:
                st.write("🎬")
                st.write("포스터 없음")

def show_api_status():
    """API 연결 상태 표시"""
    st.sidebar.title("🔧 시스템 상태")
    
    # FastAPI 상태 확인
    health_data, success = get_api_data("health")
    
    if success and health_data.get('status') == 'healthy':
        st.sidebar.markdown('<p class="status-good">✅ FastAPI 서버 정상</p>', unsafe_allow_html=True)
        
        details = health_data.get('service_details', {})
        st.sidebar.write("**서비스 상태:**")
        st.sidebar.write(f"• 모델 로드: {'✅' if details.get('model_loaded') else '❌'}")
        st.sidebar.write(f"• 데이터 로드: {'✅' if details.get('data_loaded') else '❌'}")
        st.sidebar.write(f"• 예측 완료: {'✅' if details.get('predictions_available') else '❌'}")
        st.sidebar.write(f"• 영화 수: {details.get('sample_count', 0)}개")
        
    else:
        st.sidebar.markdown('<p class="status-bad">❌ FastAPI 서버 연결 실패</p>', unsafe_allow_html=True)
        st.sidebar.error(f"오류: {health_data.get('error', '알 수 없는 오류')}")
    
    # TMDB API 상태 확인
    st.sidebar.write("**TMDB API:**")
    if TMDB_API_KEY:
        st.sidebar.markdown('<p class="status-good">✅ API 키 설정됨</p>', unsafe_allow_html=True)
        st.sidebar.write(f"API 키: {TMDB_API_KEY[:8]}***")
    else:
        st.sidebar.markdown('<p class="status-bad">⚠️ TMDB API 키 미설정</p>', unsafe_allow_html=True)
        st.sidebar.info("영화 제목과 포스터를 보려면 .env 파일에 TMDB_API_KEY를 설정하세요.")
    
    return success

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">🎬 영화 평점 예측 대시보드</h1>', unsafe_allow_html=True)
    st.markdown("**MLflow + XGBoost 기반 TMDB 영화 평점 예측 서비스**")
    st.markdown("---")
    
    # API 상태 확인
    api_connected = show_api_status()
    
    if not api_connected:
        st.error("🔌 FastAPI 서버에 연결할 수 없습니다. FastAPI 서버를 먼저 실행해주세요.")
        st.code("cd movie-mlops-project/serving\npython main.py")
        st.stop()
    
    # 사이드바 컨트롤
    st.sidebar.markdown("---")
    st.sidebar.title("🎮 옵션")
    
    # TMDB 기능 활성화 옵션
    enable_tmdb = st.sidebar.checkbox(
        "🎬 TMDB 영화 정보 표시", 
        value=True,
        help="영화 제목, 포스터, 장르 등 추가 정보 표시 (TMDB API 키 필요)"
    )
    
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (30초)", value=False)
    if auto_refresh:
        st.sidebar.write(f"마지막 업데이트: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(0.1)
        st.rerun()
    
    if st.sidebar.button("🔄 수동 새로고침"):
        st.cache_data.clear()
        st.rerun()
    
    # 메인 탭
    tab1, tab2, tab3, tab4 = st.tabs(["📊 대시보드", "🏆 TOP 영화", "📈 통계 분석", "📋 전체 데이터"])
    
    # ========================================
    # 탭 1: 대시보드
    # ========================================
    with tab1:
        st.header("📊 예측 대시보드")
        
        stats_data, stats_success = get_api_data("stats")
        
        if stats_success:
            stats = stats_data['data']['statistics']
            distribution = stats_data['data']['distribution']
            
            # 메트릭 카드
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 영화 수", f"{stats['total_movies']:,}개")
            with col2:
                st.metric("평균 예측평점", f"{stats['average_rating']:.2f}")
            with col3:
                st.metric("최고 예측평점", f"{stats['max_rating']:.2f}")
            with col4:
                st.metric("최저 예측평점", f"{stats['min_rating']:.2f}")
            
            # 예측평점 분포 차트
            st.subheader("🎯 예측평점 구간별 분포")
            
            dist_df = pd.DataFrame(list(distribution.items()), columns=['구간', '영화수'])
            
            fig_bar = px.bar(
                dist_df, 
                x='구간', 
                y='영화수',
                title="예측평점 구간별 영화 분포",
                color='영화수',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.error("통계 데이터를 불러올 수 없습니다.")
    
    # ========================================
    # 탭 2: TOP 영화 (간단한 카드 스타일)
    # ========================================
    with tab2:
        st.header("🏆 예측평점 상위 영화")
        
        # 상위 영화 개수 선택
        top_count = st.selectbox("표시할 영화 개수", [5, 10, 20, 30], index=1)
        
        top_movies_data, top_success = get_api_data("top-movies", {"limit": top_count})
        
        if top_success:
            movies = top_movies_data['data']['top_movies']
            total_movies = top_movies_data['data']['total_movies']
            
            st.info(f"총 {total_movies}개 영화 중 상위 {top_count}개 표시")
            
            # 차트 표시
            df_top = pd.DataFrame(movies)
            
            fig_top = px.bar(
                df_top, 
                x='rank', 
                y='predicted_rating',
                title=f"🎬 예측평점 상위 {top_count}개 영화",
                labels={'predicted_rating': '예측평점', 'rank': '순위'},
                color='predicted_rating',
                color_continuous_scale='plasma',
                text='predicted_rating'
            )
            fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_top.update_layout(height=500)
            st.plotly_chart(fig_top, use_container_width=True)
            
            # 영화 카드 섹션
            st.subheader("🎬 상위 영화 상세 정보")
            
            # 한 줄에 2개씩 표시 (더 깔끔하게)
            for i in range(0, len(movies[:top_count]), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(movies[:top_count]):
                        movie = movies[i + j]
                        with col:
                            # TMDB 정보 가져오기
                            movie_details = None
                            if enable_tmdb and TMDB_API_KEY:
                                movie_details = get_movie_details(movie['movie_id'])
                            
                            # 간단한 카드 표시
                            display_simple_movie_card(movie, movie_details)
            
            # 정보 메시지
            if not enable_tmdb:
                st.info("💡 TMDB 영화 정보를 보려면 사이드바에서 '🎬 TMDB 영화 정보 표시'를 체크하세요.")
            elif not TMDB_API_KEY:
                st.warning("⚠️ TMDB API 키가 설정되지 않아 기본 정보만 표시됩니다.")
        else:
            st.error("상위 영화 데이터를 불러올 수 없습니다.")
    
    # ========================================
    # 탭 3: 통계 분석
    # ========================================
    with tab3:
        st.header("📈 통계 분석")
        
        # 전체 예측 데이터 가져오기
        predictions_data, pred_success = get_api_data("predictions")
        
        if pred_success:
            results = predictions_data['data']['predictions']
            df_all = pd.DataFrame(results)
            
            st.success(f"✅ {len(df_all)}개 영화 데이터 로드 완료!")
            
            # 히스토그램
            st.subheader("📊 예측평점 분포 히스토그램")
            
            fig_hist = px.histogram(
                df_all, 
                x='predicted_rating',
                nbins=20,
                title="예측평점 분포",
                labels={'predicted_rating': '예측평점', 'count': '영화 수'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 박스플롯
            st.subheader("📦 예측평점 박스플롯")
            
            fig_box = px.box(
                df_all, 
                y='predicted_rating',
                title="예측평점 분포 (박스플롯)",
                labels={'predicted_rating': '예측평점'}
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # 통계 요약
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 기술 통계")
                desc_stats = df_all['predicted_rating'].describe()
                st.write(desc_stats)
            
            with col2:
                st.subheader("🎯 분위수 정보")
                quantiles = df_all['predicted_rating'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                st.write("**분위수별 예측평점:**")
                for q, value in quantiles.items():
                    st.write(f"• {q*100:.0f}%: {value:.2f}")
                    
        else:
            st.error("예측 데이터를 불러올 수 없습니다.")
    
    # ========================================
    # 탭 4: 전체 데이터
    # ========================================
    with tab4:
        st.header("📋 전체 예측 데이터")
        
        predictions_data, pred_success = get_api_data("predictions")
        
        if pred_success:
            results = predictions_data['data']['predictions']
            df_all = pd.DataFrame(results)
            
            # 데이터 필터링
            st.subheader("🔍 데이터 필터링")
            
            col1, col2 = st.columns(2)
            with col1:
                min_rating = st.slider("최소 예측평점", 
                                     float(df_all['predicted_rating'].min()), 
                                     float(df_all['predicted_rating'].max()), 
                                     float(df_all['predicted_rating'].min()))
            with col2:
                max_rating = st.slider("최대 예측평점", 
                                     float(df_all['predicted_rating'].min()), 
                                     float(df_all['predicted_rating'].max()), 
                                     float(df_all['predicted_rating'].max()))
            
            # 필터링된 데이터
            filtered_df = df_all[
                (df_all['predicted_rating'] >= min_rating) & 
                (df_all['predicted_rating'] <= max_rating)
            ]
            
            st.info(f"필터링 결과: {len(filtered_df)}개 영화")
            
            # 데이터 테이블
            st.subheader("📊 데이터 테이블")
            
            # 정렬 옵션
            sort_option = st.selectbox("정렬 기준", 
                                     ["예측평점 (높음→낮음)", "예측평점 (낮음→높음)", "영화 ID"])
            
            if sort_option == "예측평점 (높음→낮음)":
                filtered_df = filtered_df.sort_values('predicted_rating', ascending=False)
            elif sort_option == "예측평점 (낮음→높음)":
                filtered_df = filtered_df.sort_values('predicted_rating', ascending=True)
            else:
                filtered_df = filtered_df.sort_values('movie_id')
            
            # 순위 추가
            filtered_df_display = filtered_df.copy()
            filtered_df_display['순위'] = range(1, len(filtered_df_display) + 1)
            filtered_df_display['영화 ID'] = filtered_df_display['movie_id']
            filtered_df_display['예측평점'] = filtered_df_display['predicted_rating'].round(2)
            
            # 테이블 표시
            st.dataframe(
                filtered_df_display[['순위', '영화 ID', '예측평점']],
                use_container_width=True,
                hide_index=True
            )
            
            # CSV 다운로드
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"movie_predictions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.error("예측 데이터를 불러올 수 없습니다.")
    
    # 하단 정보
    st.markdown("---")
    st.markdown("**💡 정보**: 이 대시보드는 MLflow + XGBoost 모델을 사용하여 TMDB 영화 데이터의 평점을 예측합니다.")
    if TMDB_API_KEY:
        st.markdown("**🎬 TMDB API**: 영화 제목, 포스터, 장르 정보 연동 중")
    st.markdown(f"**🕒 마지막 업데이트**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()