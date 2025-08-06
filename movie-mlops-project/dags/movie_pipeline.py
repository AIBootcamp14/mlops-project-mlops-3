from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import logging
import subprocess
import sys


# DAG 기본 설정값 정의
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

# 프로젝트 루트 경로
project_root = "/mnt/c/Users/can40/mlops-project-mlops-3/movie-mlops-project"


# DAG 객체 생성
dag = DAG(
    'movie_pipeline_all_scripts',
    default_args=default_args,
    schedule_interval=timedelta(minutes=5), # 5분마다 실행
    catchup=False, # 과거 실행 태스크 몰아서 실행 x
    max_active_runs=1, # 동시에 최대 1개
    description='7개 스크립트를 순차 실행하는 영화 평점 예측 파이프라인',
)


# 실제 스크립트 실행 함수 정의
def run_script(script_name, env_vars=None):
    # 실행할 스크립트 전체 경로 생성
    script_path = os.path.join(project_root, script_name)

    # 스크립트 파일 존재 확인
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} 가 존재하지 않습니다.")

    logging.info(f"{script_name} 실행 시작")

    # 환경 변수 복사
    env = os.environ.copy()

    # 추가로 전달받은 환경변수 업데이트
    if env_vars:
        env.update(env_vars)
    

    # subprocess를 이용해 파이썬 인터프리터로 해당 스크립트를 실행
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
    )

    # 실행 결과 로그 출력
    logging.info(f"{script_name} stdout:\n{result.stdout}")

    # 반환 코드가 0 아니면 에러 발생
    if result.returncode != 0:
        logging.error(f"{script_name} 실행 실패 stderr:\n{result.stderr}")
        raise RuntimeError(f"{script_name} 실행 중 오류 발생, 반환 코드: {result.returncode}")
    logging.info(f"{script_name} 실행 완료")


# 크롤러 스크립트는 API 키 등 환경변수가 필요해 별도 함수로 분리
def run_crawler():
    return run_script('preprocessing/crawler.py', env_vars={
        "TMDB_API_KEY": "4cb727de9fdb0d2cf868b2c31ab39e93",
        "TMDB_BASE_URL": "https://api.themoviedb.org/3/movie",
    })


# Airflow task 정의 - 크롤러 실행 task
crawl_task = PythonOperator(
    task_id='run_crawler',
    python_callable=run_crawler,
    dag=dag,
    
)


# 그 외 스크립트 단순히 run_script 함수 호출
preprocess_task = PythonOperator(
    task_id='run_preprocessing',
    python_callable=lambda: run_script('preprocessing/preprocessing.py'),
    dag=dag,
)

train_task = PythonOperator(
    task_id='run_train',
    python_callable=lambda: run_script('model/train.py'),
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='run_evaluate',
    python_callable=lambda: run_script('model/evaluate.py'),
    dag=dag,
)

register_task = PythonOperator(
    task_id='run_register_mlflow',
    python_callable=lambda: run_script('model/register_mlflow.py'),
    dag=dag,
)

main_task = PythonOperator(
    task_id='run_main',
    python_callable=lambda: run_script('preprocessing/main.py'),
    dag=dag,
)

# utils.py는 라이브러리 역할이므로 실행 작업 필요 없음

# 실행 순서 지정
crawl_task >> preprocess_task >> train_task >> evaluate_task >> register_task >> main_task


# 수동 실행 

# 1. Airflow 웹 UI에서 수동 실행

# 2. CLI에서 수동 실행
# (1) DAG 전체 실행
# airflow dags trigger movie_pipeline_all_scripts
# (2) 특정 실행 시간 지정 실행
# airflow dags trigger -e 2025-08-05T12:00:00 movie_pipeline_all_scripts
# (3) 특정 태스크만 실행
# airflow tasks run movie_pipeline_all_scripts run_train 2025-08-05T12:00:00 (run_train)

# 3. CLI 테스트 모드 실행 (스케줄러 거치지 않고 실행)
# airflow dags test movie_pipeline_all_scripts 2025-08-05T12:00:00


#  왜 코드에 따로 수동 트리거 함수를 넣지 않는가?
# Airflow의 trigger는 Airflow 명령어로 DAG 실행 요청을 보내는 것이지, DAG 내부에서 호출하는 함수가 아님
# DAG 코드에 수동 실행 함수를 넣는다면 직접 태스크를 강제로 돌리는 코드가 들어가야 함