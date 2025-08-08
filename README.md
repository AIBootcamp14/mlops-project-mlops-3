# 🎬 영화 평점 예측 서비스

> **목표:** 영화 평점 예측 서비스를 자동화하는 **MLOps 파이프라인 구현**  
> **배경:** MLOps 전 과정을 직접 구현하여 실무 역량 강화  

- **프로젝트 기간:** 2025.07.28 ~ 2025.08.08  
- **배포 링크:** [서비스 바로가기](http://54.180.90.11:8501)

---

## 📦 서비스 구성 요소

### 1. 주요 기능

- TMDB API를 이용한 영화 메타데이터 수집
- 수집 데이터 전처리 및 피처 엔지니어링
- XGBoost 기반 회귀 모델 학습 및 예측
- FastAPI를 이용한 API 제공 및 Streamlit 시각화
- Airflow를 이용한 파이프라인 자동화
- MLflow를 통한 실험 및 모델 버전 관리

### 2. 사용자 흐름

1. 사용자가 영화 제목 또는 ID 입력  
2. FastAPI를 통해 입력 전달  
3. MLflow에서 서빙 중인 모델 호출  
4. 예측 결과 반환 및 Streamlit에서 시각화  

---

## ⚙ 활용 장비 및 협업 툴

### 서버 및 개발 장비

- **서버:** AWS EC2 t2.medium  
- **개발환경:** Ubuntu 20.04 (WSL2)  
- **테스트:** 개인 PC  

### 협업 툴

- **소스 관리:** GitHub (Issue & PR 중심)  
- **프로젝트 관리:** Notion  
- **커뮤니케이션:** Slack, KakaoTalk  

---

## 🧠 AI 모델 개요

- **모델 이름:** XGBoost (eXtreme Gradient Boosting)
- **선정 이유:**  
  - 수치 예측(회귀)에 최적화  
  - 빠른 학습 속도 및 과적합 방지 기법(L1/L2, Shrinkage 등)  
  - 향후 특성 중요도 해석 가능성  

- **학습 데이터:**  
  - TMDB API 기반 영화 메타데이터  
  - 감독, 배우, 장르, 출시일 등 주요 특성 포함

- **모델 성능 평가 지표:**  
  - RMSE, MAE, R² (MLflow UI에서 확인 가능)

---

## 🛠 아키텍처 및 워크플로우

### 시스템 구조도
<img src="https://github.com/user-attachments/assets/49132d70-e853-49f6-b598-ec42f2c69726" width="100%"/>

### MLOps 워크플로우 (Airflow DAG)

- `run_crawler` → 영화 메타데이터 수집  
- `run_preprocessing` → 데이터 정제 및 저장  
- `run_train` → 모델 학습  
- `run_evaluate` → 평가 지표 산출  
- `run_register_mlflow` → 모델 MLflow 등록  
- `run_main` → 전체 파이프라인 통합  

---

## 🚀 기술 스택

### 백엔드
- **FastAPI** (API 서버)
- **MLflow** (모델 추적 및 배포)
- **Amazon S3** (모델 아티팩트 및 전처리 데이터 저장)

### 프론트엔드
- Streamlit (사용자 대시보드)

### 머신러닝
- XGBoost, scikit-learn, pandas, NumPy

### DevOps & 배포
- Docker, Airflow, AWS EC2

---

## **6. 팀원 소개**  


| <img src="https://avatars.githubusercontent.com/u/213385368?v=4" width="150" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/66048976?v=4" width="150" style="border-radius:50%;">  | <img src="https://avatars.githubusercontent.com/u/162023876?v=4" width="150" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/213417897?v=4" width="150" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/213385147?v=4" width="150" style="border-radius:50%;"> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [문채린](https://github.com/CHAERINMOON)             |            [박재홍](https://github.com/woghd8503)             |            [김상윤](https://github.com/94KSY)             |            [김동준](https://github.com/rafiki3816)             |            [정서우](https://github.com/Seowoo-C)             |
|                            팀장, 자동화 파이프라인 구축                             |                            개발 환경 세팅, 아키텍처 설계                             |                            웹서비스 구현                             |                            데이터 크롤링 전처리                            |                            모델 학습, MLflow                             |
> **협업 철학:**  
> “배워서 남주자”, “모든 질문은 가치가 있다”, “모든 새로운 시도는 가치가 있다”  
> 매일 오전 10시 미팅, 오후 2시 집중 회의 진행.

---

# 📝 팀원 회고 요약

|   이름           | 역할                         | 회고 요약 |
|------------------|------------------------------|-----------|
|  **문채린**          | 자동화 파이프라인 구축       | 자동화 파트를 처음 맡아 걱정이 있었지만 파이프라인 구축 방법을 익히며 실무 역량 향상. 팀원들의 도움에 감사. |
|  **박재홍**          | 아키텍처 설계, 개발환경 세팅 | 공동 성장을 목표로 다양한 시도를 하며 많은 것을 배움. 다음에는 더 주도적으로 참여하고자 다짐. |
|  **김상윤**           | 웹서비스 구현                | 다른 파트와의 연결에서 어려움을 겪었지만 많은 배움을 얻음. 과정의 중요성을 깨달음. |
|  **김동준**           | 데이터 크롤링, 전처리        | 단계별 협업이 어려웠지만 문제를 공유하며 협업 능력 향상. 모두의 고생에 감사함. |
|  **정서우**           | 모델 학습, MLflow            | 처음 접하는 도구들이 어려웠지만, 팀원 도움으로 무사히 역할 수행. 흐름을 이해할 수 있었던 소중한 경험. |


## **7. Appendix**  
### **7.1 참고 자료**  
- TMDB API 공식 문서: https://developer.themoviedb.org/docs
- XGBoost 공식 문서: https://xgboost.readthedocs.io
- MLflow 사용 가이드: https://mlflow.org/docs/latest/index.html
- Streamlit 문서: https://docs.streamlit.io/

### **7.2 설치 및 실행 방법**  
1. **필수 라이브러리 설치:**  
    ```bash
    pip install -r requirements.txt
    ```

2. **서버 실행:**  
    ```bash
    python app.py
    ```

3. **웹페이지 접속:**  
    ```
    http://localhost:5000
    ```

### **7.4 주요 실행 주소**  
| 서비스 | 주소 |
|--------|------|
| :globe_with_meridians: MLflow UI | [http://54.180.90.11:5001](http://54.180.90.11:5001) |
| :satellite_antenna: FastAPI Docs | [http://54.180.90.11:8000/docs](http://54.180.90.11:8000/docs) |
| :bar_chart: Streamlit Dashboard | [http://54.180.90.11:8501](http://54.180.90.11:8501) |


