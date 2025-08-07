import logging
import os
import sys
import argparse
from typing import Optional

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

from model.utils import get_config_value

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_best_run_id(client: MlflowClient, experiment_name: str, metric_name: str = "rmse") -> str:
    """
    MLflow Tracking API를 사용하여 주어진 실험에서 가장 낮은 지표를 가진 Run의 ID를 반환합니다.
    """
    logging.info(f"실험 '{experiment_name}'에서 최적의 Run ID를 찾고 있습니다.")
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"실험 '{experiment_name}'을 찾을 수 없습니다.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{metric_name} ASC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"실험 '{experiment_name}'에 유효한 Run이 없습니다.")

    best_run = runs[0]
    logging.info(f"최적의 Run을 찾았습니다. Run ID: {best_run.info.run_id}, {metric_name}: {best_run.data.metrics[metric_name]:.4f}")
    
    return best_run.info.run_id

def register_and_promote_model(best_run_id: str,
                                model_registry_name: str,
                                stage: str = "Production"):
    """
    주어진 Run ID의 모델을 MLflow 모델 레지스트리에 등록하고 특정 스테이지로 전환합니다.
    """
    logging.info(f"최적 Run ID '{best_run_id}'의 모델을 레지스트리에 등록하고 '{stage}' 스테이지로 전환합니다.")

    # MLflow에 저장된 모델 아티팩트 URI
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        # ✅ MLflow 모델 레지스트리에 등록
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name
        )
        
        logging.info(
            f"모델 레지스트리에 등록 완료. 이름: {registered_model.name}, 버전: {registered_model.version}"
        )

        client = MlflowClient()
        
        # ✅ 모델 레지스트리 스테이지 전환
        # 기존에 Production에 있던 모델이 있다면, None으로 변경
        for mv in client.search_model_versions(f"name='{model_registry_name}'"):
            if mv.current_stage == stage:
                client.transition_model_version_stage(
                    name=model_registry_name,
                    version=mv.version,
                    stage="None"
                )
                logging.info(f"이전 '{stage}' 모델 (버전: {mv.version})의 스테이지를 'None'으로 변경했습니다.")

        # 새로 등록된 모델을 Production 스테이지로 변경
        client.transition_model_version_stage(
            name=registered_model.name,
            version=registered_model.version,
            stage=stage
        )

        logging.info(
            f"새로운 모델 (버전: {registered_model.version})을 '{stage}' 스테이지로 전환했습니다."
        )
        
    except Exception as e:
        logging.error(f"모델 등록 및 스테이지 전환 중 치명적인 오류 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="최적의 모델을 MLflow에 등록하고 Production 스테이지로 전환하는 스크립트")
    parser.add_argument("--run_id", type=str, help="등록할 MLflow Run ID (지정하지 않으면 최적의 Run을 자동으로 찾음)", default=None)
    args = parser.parse_args()

    try:
        # ✅ config 경로 설정 및 값 로딩
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        mlflow_tracking_uri = get_config_value(config_path, 'mlflow.tracking_uri')
        mlflow_experiment_name = get_config_value(config_path, 'mlflow.experiment_name')
        model_registry_name = get_config_value(config_path, 'mlflow.model_registry_name')

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient()

        # ✅ 등록할 Run ID 결정
        run_id_to_register = args.run_id
        if run_id_to_register is None:
            logging.info("Run ID가 지정되지 않아 최적의 Run을 자동으로 찾습니다.")
            run_id_to_register = find_best_run_id(client, mlflow_experiment_name, "rmse")

        # ✅ 모델 등록 및 스테이지 전환 실행
        register_and_promote_model(
            best_run_id=run_id_to_register,
            model_registry_name=model_registry_name,
            stage="Production"
        )
        
        logging.info("✅ 모델 등록 및 스테이지 전환 완료.")

    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)