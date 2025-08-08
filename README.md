# 🎬 영화 평점 예측 서비스

> **목표:** 영화 평점 예측 서비스를 자동화하는 **MLOps 파이프라인 구현**  
> **배경:** MLOps 전 과정을 직접 구현하여 실무 역량 강화  

- **프로젝트 기간:** 2025.07.28 ~ 2025.08.08  
- **배포 링크:** [Streamlit 대시보드 바로가기](http://54.180.90.11:8501)

---

## 🔧 서비스 구성

### ✅ 주요 기능

- TMDB API 기반 영화 메타데이터 수집
- 수집 데이터 전처리 및 Feature Engineering
- XGBoost 기반 회귀 모델 학습 및 예측
- FastAPI를 활용한 REST API 제공
- Streamlit을 통한 예측 결과 시각화
- Airflow 기반 파이프라인 자동화
- MLflow를 통한 실험 및 모델 버전 관리

### ✅ 사용자 흐름

1. 사용자가 영화 제목 또는 ID 입력  
2. FastAPI를 통해 요청 전달  
3. MLflow 모델 호출 및 예측  
4. Streamlit을 통해 결과 시각화  

---

## 🧠 AI 모델 정보

- **모델 이름:** XGBoost (eXtreme Gradient Boosting)
- **선정 이유:**
  - 회귀 문제에 적합한 고성능 예측
  - 빠른 학습 및 과적합 방지 기법 지원
  - 향후 특성 중요도 해석 가능

- **입력 데이터:**  
  - TMDB API로 수집한 영화 메타데이터 (감독, 배우, 장르, 개봉일 등)

- **성능 지표:**  
  - RMSE, MAE, R² (MLflow UI로 확인)

---

## 🛠 아키텍처 및 파이프라인

### ✅ 시스템 구조도
<img src="https://github.com/user-attachments/assets/49132d70-e853-49f6-b598-ec42f2c69726" width="100%"/>

### ✅ MLOps 워크플로우 (Airflow DAG)

- `run_crawler` → 영화 데이터 수집  
- `run_preprocessing` → 전처리 수행  
- `run_train` → 모델 학습  
- `run_evaluate` → 성능 평가  
- `run_register_mlflow` → MLflow 모델 등록  
- `run_main` → 전체 파이프라인 실행  

---

## 📁 폴더 구조

```
movie-mlops-project/
├── .github/workflows/          # GitHub Actions CI/CD
├── config/                     # 설정 파일 (config.yml 등)
├── dags/                       # Airflow DAG 정의
├── docker/                     # Docker 관련 파일
├── ingestion/                  # TMDB API 데이터 수집
├── model/                      # 모델 학습 및 평가
├── model_artifacts/            # 저장된 모델 파일
├── preprocessing/              # 전처리 및 Feature Engineering
├── requirements/               # 의존성 설정
├── scripts/                    # 실행 스크립트
├── serving/                    # FastAPI 서버 구성
├── tests/                      # 유닛 테스트
├── .env.template               # 환경변수 예시
├── docker-compose.yml          # Docker 통합 실행 파일
├── pyproject.toml              # 프로젝트 메타 정보
└── requirements.txt            # 패키지 목록
```

---

## 🚀 기술 스택

### ✅ 백엔드
- **FastAPI** – API 서버  
- **MLflow** – 모델 추적 및 배포  
- **Amazon S3** – 모델 및 데이터 저장  

### ✅ 프론트엔드
- **Streamlit** – 대시보드 UI  

### ✅ 머신러닝
- **XGBoost**, **scikit-learn**, **pandas**, **NumPy**  

### ✅ DevOps & MLOps
- **Docker**, **Airflow**, **AWS EC2**, **GitHub Actions**

---

## 👥 팀원 소개

| <img src="https://avatars.githubusercontent.com/u/213385368?v=4" width="120" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/66048976?v=4" width="120" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/162023876?v=4" width="120" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/213417897?v=4" width="120" style="border-radius:50%;"> | <img src="https://avatars.githubusercontent.com/u/213385147?v=4" width="120" style="border-radius:50%;"> |
|:---:|:---:|:---:|:---:|:---:|
| [문채린](https://github.com/CHAERINMOON) | [박재홍](https://github.com/woghd8503) | [김상윤](https://github.com/94KSY) | [김동준](https://github.com/rafiki3816) | [정서우](https://github.com/Seowoo-C) |
| 팀장, 자동화 파이프라인 구축 | 아키텍처 설계 및 환경 세팅 | 웹서비스 구현 | 데이터 수집 및 전처리 | 모델 학습 및 MLflow |

> **협업 철학:**  
> "배워서 남주자" / "모든 질문은 가치가 있다" / "모든 새로운 시도는 가치가 있다"  
> 매일 오전 10시 미팅, 오후 2시 집중 회의 진행

---

## 💬 팀원 회고 요약

| 이름 | 역할 | 회고 요약 |
|------|------|-----------|
| **문채린** | 자동화 파이프라인 구축 | 처음 맡은 자동화 파트에 대한 도전과 성공 경험. 팀원의 도움에 감사. |
| **박재홍** | 아키텍처 및 환경 구성 | 공동 성장 중심으로 협업하며 많은 것을 시도. 다음 프로젝트엔 더 주도적으로 참여 예정. |
| **김상윤** | 웹서비스 구축 | 예상보다 많은 연결 문제가 있었지만 큰 배움과 책임감을 얻은 경험. |
| **김동준** | 데이터 전처리 | 협업 과정의 어려움을 극복하며 모두와 함께 성장한 값진 경험. |
| **정서우** | 모델링, MLflow | 낯선 도구 속에서도 팀원 도움으로 흐름을 익히고 MLOps 이해도 향상. |

---

## 🧪 실행 방법

1. **라이브러리 설치**
```bash
pip install -r requirements.txt
```

2. **FastAPI 서버 실행**
```bash
python app.py
```

3. **접속 주소**
```
http://localhost:5000
```

---

## 🔗 주요 실행 주소

| 서비스 항목 | 접속 주소 |
|-------------|-----------|
| 🌐 MLflow UI | http://54.180.90.11:5001 |
| 📡 FastAPI Docs | http://54.180.90.11:8000/docs |
| 📊 Streamlit Dashboard | http://54.180.90.11:8501 |

---

## 📚 참고 자료

- [TMDB API 공식 문서](https://developer.themoviedb.org/docs)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io)
- [MLflow 문서](https://mlflow.org/docs/latest/index.html)
- [Streamlit 문서](https://docs.streamlit.io/)
