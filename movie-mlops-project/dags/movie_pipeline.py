from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
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
    schedule_interval=None,
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
