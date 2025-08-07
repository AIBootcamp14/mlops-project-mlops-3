import logging
import os
import sys
import argparse

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np

from dotenv import load_dotenv

# ✅ 루트 경로 설정 및 환경 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅ .env 로드 (AWS 자격 정보용)
load_dotenv() 

from model.utils import evaluate_model, get_config_value, load_data, load_feature_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_and_log_model(run_id: str,
                           test_filepath: str,
                           feature_names: list,
                           target_column: str,
                           mlflow_tracking_uri: str,
                           mlflow_experiment_name: str) -> dict:
    """
    MLflow Run ID를 기반으로 모델을 로드하고, 테스트 데이터로 평가한 후
    해당 Run에 지표를 로깅합니다.
    """
    logging.info(f"MLflow Run ID: {run_id}의 모델을 평가합니다.")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # MLflow에 저장된 모델 아티팩트 URI
    model_uri = f"runs:/{run_id}/model"
    
    with mlflow.start_run(run_id=run_id) as run:
        logging.info(f"MLflow Run '{run.info.run_id}'에 재참여하여 모델 평가 시작")
        
        try:
            # ✅ MLflow 아티팩트에서 모델 로드
            model = mlflow.xgboost.load_model(model_uri)
            logging.info(f"MLflow 아티팩트에서 모델 로드 완료: {model_uri}")
            
            test_df = load_data(test_filepath)
            X_test = test_df[feature_names]
            y_test = test_df[target_column]

            logging.info(f"테스트 데이터 X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)

            mlflow.log_metrics(metrics)
            logging.info(f"MLflow에 평가 지표 로깅 완료: {metrics}")

            return metrics

        except Exception as e:
            logging.error(f"모델 평가 중 오류 발생: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 평가 스크립트")
    parser.add_argument("--run_id", type=str, help="평가할 MLflow Run ID (지정하지 않으면 최신 Run을 사용)", default=None)
    args = parser.parse_args()

    try:
        # ✅ config 경로 설정 및 값 로딩
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        processed_data_dir = get_config_value(config_path, 'data.processed_data_dir')
        test_data_file = get_config_value(config_path, 'data.test_data_file')
        feature_names_file = get_config_value(config_path, 'data.feature_names_file')
        target_column = get_config_value(config_path, 'model.target_column')
        mlflow_tracking_uri = get_config_value(config_path, 'mlflow.tracking_uri')
        mlflow_experiment_name = get_config_value(config_path, 'mlflow.experiment_name')

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # ✅ 평가할 Run ID 결정
        run_id_to_evaluate = args.run_id
        if run_id_to_evaluate is None:
            logging.info("Run ID가 지정되지 않아 최신 Run을 자동으로 찾습니다.")
            experiment = client.get_experiment_by_name(mlflow_experiment_name)
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if runs:
                    run_id_to_evaluate = runs[0].info.run_id
                    logging.info(f"최신 Run ID: {run_id_to_evaluate}를 사용합니다.")
                else:
                    logging.error("지정된 실험에 Run이 존재하지 않습니다.")
                    sys.exit(1)
            else:
                logging.error(f"실험 '{mlflow_experiment_name}'을 찾을 수 없습니다.")
                sys.exit(1)

        # ✅ 모델 평가 실행
        metrics = evaluate_and_log_model(
            run_id=run_id_to_evaluate,
            test_filepath=os.path.join(project_root, processed_data_dir, test_data_file),
            feature_names=load_feature_names(os.path.join(project_root, processed_data_dir, feature_names_file)),
            target_column=target_column,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name
        )
        
        logging.info(f"✅ 모델 평가 완료. 결과: {metrics}")

    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)