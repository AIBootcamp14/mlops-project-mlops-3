import logging
import os
import sys
import argparse

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv

# ✅ 루트 경로 설정 및 환경 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅ .env 로드 (AWS 자격 정보용)
load_dotenv()

from model.utils import get_config_value, save_model, load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_best_run_id(client: MlflowClient, experiment_name: str, metric_name: str = "rmse") -> str:
    """
    MLflow Tracking API를 사용하여 주어진 실험에서 가장 낮은 지표를 가진 Run의 ID를 반환합니다.
    """
    logging.info(f"실험 '{experiment_name}'에서 최적의 Run ID를 찾고 있습니다.")
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"실험 '{experiment_name}'을 찾을 수 없습니다.")

    # 가장 낮은 지표를 가진 Run을 검색
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{metric_name} ASC"],  # RMSE는 낮을수록 좋으므로 ASC (오름차순)
        max_results=1
    )

    if not runs:
        raise ValueError(f"실험 '{experiment_name}'에 유효한 Run이 없습니다.")

    best_run = runs[0]
    logging.info(f"최적의 Run을 찾았습니다. Run ID: {best_run.info.run_id}, {metric_name}: {best_run.data.metrics[metric_name]:.4f}")
    
    return best_run.info.run_id

def register_and_save_model(best_run_id: str,
                            model_registry_name: str,
                            model_output_dir: str,
                            model_filename: str):
    """
    주어진 Run ID의 모델을 MLflow 모델 레지스트리에 등록하고 로컬에 저장합니다.
    """
    logging.info(f"최적 Run ID '{best_run_id}'의 모델을 레지스트리에 등록 및 로컬에 저장합니다.")

    # MLflow에 저장된 모델 아티팩트 URI
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        # ✅ MLflow 모델 레지스트리에 등록
        # mlflow.register_model 함수는 모델 아티팩트 URI를 통해 등록을 수행
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name
        )
        
        logging.info(
            f"모델 레지스트리에 등록 완료. 이름: {registered_model.name}, 버전: {registered_model.version}"
        )

        # ✅ 로컬 파일 시스템에 모델 저장
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_save_path = os.path.join(project_root, model_output_dir, model_filename)

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # 아티팩트에서 모델을 로드한 후 로컬에 저장
        model_to_save = mlflow.xgboost.load_model(model_uri)
        save_model(model_to_save, model_save_path)
        
        logging.info(f"모델을 로컬에 저장했습니다: {model_save_path}")

    except Exception as e:
        logging.error(f"모델 등록 및 저장 중 치명적인 오류 발생: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="최적의 모델을 MLflow에 등록하고 로컬에 저장하는 스크립트")
    parser.add_argument("--run_id", type=str, help="등록할 MLflow Run ID (지정하지 않으면 최적의 Run을 자동으로 찾음)", default=None)
    args = parser.parse_args()

    try:
        # ✅ config 경로 설정 및 값 로딩
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        mlflow_tracking_uri = get_config_value(config_path, 'mlflow.tracking_uri')
        mlflow_experiment_name = get_config_value(config_path, 'mlflow.experiment_name')
        model_registry_name = get_config_value(config_path, 'mlflow.model_registry_name')
        model_output_dir = get_config_value(config_path, 'model.output_dir')
        model_filename = get_config_value(config_path, 'model.filename')

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient()

        # ✅ 등록할 Run ID 결정
        run_id_to_register = args.run_id
        if run_id_to_register is None:
            logging.info("Run ID가 지정되지 않아 최적의 Run을 자동으로 찾습니다.")
            run_id_to_register = find_best_run_id(client, mlflow_experiment_name, "rmse")

        # ✅ 모델 등록 및 저장 실행
        register_and_save_model(
            best_run_id=run_id_to_register,
            model_registry_name=model_registry_name,
            model_output_dir=model_output_dir,
            model_filename=model_filename
        )
        
        logging.info("✅ 모델 등록 및 로컬 저장 완료.")

    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)