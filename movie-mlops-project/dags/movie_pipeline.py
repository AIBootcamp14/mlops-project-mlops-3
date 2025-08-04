from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 1) 기본 설정: DAG 인스턴스 생성
default_args = {
    'owner': 'can40',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_ml_pipeline',
    default_args=default_args,
    description='ML 파이프라인 자동화 예제',
    schedule_interval='@daily',  # 매일 실행
    catchup=False,  # 과거 실행 건 무시
)

# 2) 실행할 파이썬 함수 정의 (예: 데이터 전처리, 학습, 평가)
def preprocess():
    print("데이터 전처리 중...")

def train():
    print("모델 학습 중...")

def evaluate():
    print("모델 평가 중...")

# 3) PythonOperator로 태스크(task) 생성
task_preprocess = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train',
    python_callable=train,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate',
    python_callable=evaluate,
    dag=dag,
)

# 4) 태스크 간 순서 지정
task_preprocess >> task_train >> task_evaluate

