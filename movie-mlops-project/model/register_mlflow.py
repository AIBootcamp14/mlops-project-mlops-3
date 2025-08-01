import logging
import os

import mlflow
import mlflow.xgboost

from model.utils import get_config_value, load_model, save_model

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def register_and_save_model(model, mlflow_run_id: str):
    """
    학습된 모델을 MLflow에 아티팩트로 저장하고 모델 레지스트리에 등록하며,
    로컬 파일 시스템에도 저장합니다.
    """
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml'
        )

        model_output_dir = get_config_value(config_path, 'model.output_dir')
        model_filename = get_config_value(config_path, 'model.filename')
        mlflow_tracking_uri = get_config_value(config_path, 'mlflow.tracking_uri')
        model_registry_name = get_config_value(config_path, 'mlflow.model_registry_name')

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Movie Rating Prediction")

        with mlflow.start_run(run_id=mlflow_run_id) as run:
            logging.info(f"MLflow Run '{run.info.run_id}'에 재참여하여 모델 등록 및 저장 시작")

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="xgboost-model",
                registered_model_name=model_registry_name
            )
            logging.info(
                f"MLflow에 모델 아티팩트 '{model_registry_name}' 로깅 및 레지스트리 등록 완료"
            )

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_save_path = os.path.join(project_root, model_output_dir, model_filename)

            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            save_model(model, model_save_path)
            logging.info(f"모델을 로컬에 저장했습니다: {model_save_path}")

    except Exception as e:
        logging.error(f"모델 등록 및 저장 중 치명적인 오류 발생: {e}", exc_info=True)
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

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, model_output_dir, model_filename)

        if os.path.exists(model_path):
            trained_model = load_model(model_path)
            with mlflow.start_run(run_name="Register_Standalone_Test") as temp_run:
                register_and_save_model(trained_model, temp_run.info.run_id)
                logging.info("단독 실행 모델 등록 및 저장 완료.")
        else:
            logging.error(
                f"모델 파일이 없습니다: {model_path}. 먼저 train.py를 실행하여 모델을 학습시키세요."
            )
    except Exception as e:
        logging.error(f"단독 실행 모델 등록 및 저장 중 오류 발생: {e}")