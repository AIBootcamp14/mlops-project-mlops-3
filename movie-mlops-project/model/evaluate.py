import logging
import os

import mlflow
import numpy as np
import pandas as pd

from model.utils import evaluate_model, get_config_value, load_data, load_feature_names, load_model

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def evaluate_and_log_model(model, mlflow_run_id: str):
    """
    주어진 모델을 테스트 데이터로 평가하고 MLflow에 지표를 로깅합니다.
    """
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml'
        )

        processed_data_dir = get_config_value(config_path, 'data.processed_data_dir')
        test_data_file = get_config_value(config_path, 'data.test_data_file')
        feature_names_file = get_config_value(config_path, 'data.feature_names_file')
        target_column = get_config_value(config_path, 'model.target_column')
        mlflow_tracking_uri = get_config_value(config_path, 'mlflow.tracking_uri')

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Movie Rating Prediction")

        with mlflow.start_run(run_id=mlflow_run_id) as run:
            logging.info(f"MLflow Run '{run.info.run_id}'에 재참여하여 모델 평가 시작")

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_filepath = os.path.join(
                project_root, processed_data_dir, test_data_file
            )
            feature_names_filepath = os.path.join(
                project_root, processed_data_dir, feature_names_file
            )

            test_df = load_data(test_filepath)
            feature_names = load_feature_names(feature_names_filepath)

            X_test = test_df[feature_names]
            y_test = test_df[target_column]

            logging.info(
                f"테스트 데이터 X_test shape: {X_test.shape}, y_test shape: {y_test.shape}"
            )

            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)

            mlflow.log_metrics(metrics)
            logging.info("MLflow에 평가 지표 로깅 완료")

            return metrics

    except Exception as e:
        logging.error(f"모델 평가 중 치명적인 오류 발생: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.warning(
        "단독 실행: 테스트 모델 로드. 주로 파이프라인에서 호출됩니다."
    )

    try:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml'
        )
        model_output_dir = get_config_value(config_path, 'model.output_dir')
        model_filename = get_config_value(config_path, 'model.filename')
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            model_output_dir,
            model_filename,
        )

        if os.path.exists(model_path):
            trained_model = load_model(model_path)
            with mlflow.start_run(run_name="Evaluate_Standalone_Test") as temp_run:
                metrics = evaluate_and_log_model(trained_model, temp_run.info.run_id)
                logging.info(f"단독 실행 평가 완료. 결과: {metrics}")
        else:
            logging.error(
                f"모델 파일이 없습니다: {model_path}. 먼저 train.py를 실행하여 모델을 학습시키세요."
            )
    except Exception as e:
        logging.error(f"단독 실행 평가 중 오류 발생: {e}")