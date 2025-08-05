from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import logging
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dag = DAG(
    'movie_pipeline_all_scripts',
    default_args=default_args,
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1,
    description='7개 스크립트를 순차 실행하는 영화 평점 예측 파이프라인',
)

def run_script(script_name):
    script_path = os.path.join(project_root, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} 가 존재하지 않습니다.")

    logging.info(f"{script_name} 실행 시작")
    result = subprocess.run(
        ['python3', script_path],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    logging.info(f"{script_name} stdout:\n{result.stdout}")
    if result.returncode != 0:
        logging.error(f"{script_name} 실행 실패 stderr:\n{result.stderr}")
        raise RuntimeError(f"{script_name} 실행 중 오류 발생, 반환 코드: {result.returncode}")
    logging.info(f"{script_name} 실행 완료")

crawl_task = PythonOperator(
    task_id='run_crawler',
    python_callable=lambda: run_script('crawler.py'),
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='run_preprocessing',
    python_callable=lambda: run_script('preprocessing.py'),
    dag=dag,
)

train_task = PythonOperator(
    task_id='run_train',
    python_callable=lambda: run_script('train.py'),
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='run_evaluate',
    python_callable=lambda: run_script('evaluate.py'),
    dag=dag,
)

register_task = PythonOperator(
    task_id='run_register_mlflow',
    python_callable=lambda: run_script('register_mlflow.py'),
    dag=dag,
)

main_task = PythonOperator(
    task_id='run_main',
    python_callable=lambda: run_script('main.py'),
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