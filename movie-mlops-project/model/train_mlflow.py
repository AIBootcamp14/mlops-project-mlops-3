import logging
import os
import sys
import argparse
import json

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv

# ✅ 루트 경로 설정 및 환경 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅ .env 로드 (AWS 자격 정보용)
load_dotenv()

from model.utils import get_config_value, load_data, load_feature_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(
    train_filepath: str,
    target_column: str,
    feature_names: list,
    xgb_params: dict,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str
) -> None:
    """
    XGBoost 모델을 학습하고 MLflow에 로깅합니다.
    """
    logging.info("모델 학습을 시작합니다.")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    run_name = f"XGBoost_Train_MaxDepth_{xgb_params.get('max_depth', 'default')}_Eta_{xgb_params.get('eta', 'default')}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}으로 모델 학습 시작")

        try:
            train_df = load_data(train_filepath)

            # ✅ 모든 컬럼 이름을 문자열로 강제 변환
            train_df.columns = train_df.columns.astype(str)

            # ✅ 디버깅: feature_names와 컬럼 비교
            missing = [f for f in feature_names if f not in train_df.columns]
            if missing:
                logging.error(f"❗ train_df에 존재하지 않는 feature들: {missing}")
                logging.info(f"💡 train_df 컬럼 목록: {train_df.columns.tolist()}")
                raise KeyError(f"{missing} not in train_df.columns")

            X_train = train_df[feature_names]
            y_train = train_df[target_column]

            logging.info(f"학습 데이터 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            mlflow.log_params(xgb_params)
            logging.info(f"MLflow에 파라미터 로깅 완료: {xgb_params}")

            model = xgb.XGBRegressor(**xgb_params)
            logging.info("XGBoost 모델 학습 시작")
            model.fit(X_train, y_train)
            logging.info("XGBoost 모델 학습 완료")

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
            )
            logging.info("MLflow에 모델 아티팩트 로깅 완료 (레지스트리 등록 제외)")

            return

        except Exception as e:
            logging.error(f"모델 학습 중 오류 발생: {e}", exc_info=True)
            mlflow.end_run(status="FAILED")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost 모델 학습 스크립트")
    parser.add_argument("--max_depth", type=int, help="XGBoost max_depth 파라미터 (기본값을 오버라이드)")
    parser.add_argument("--eta", type=float, help="XGBoost eta (learning_rate) 파라미터 (기본값을 오버라이드)")
    args = parser.parse_args()

    try:
        # ✅ config 경로 설정
        config_path = os.path.join(project_root, 'config', 'config.yaml')

        # ✅ config 값 로딩
        processed_data_dir = get_config_value(config_path, 'data.processed_data_dir')
        train_data_file = get_config_value(config_path, 'data.train_data_file')
        feature_names_file = get_config_value(config_path, 'data.feature_names_file')
        target_column = get_config_value(config_path, 'model.target_column')
        base_xgb_params = get_config_value(config_path, 'model.xgboost_params')

        mlflow_tracking_uri = get_config_value(config_path, 'mlflow.tracking_uri')
        mlflow_experiment_name = get_config_value(config_path, 'mlflow.experiment_name')

        # ✅ 명령줄 인자 반영
        current_xgb_params = base_xgb_params.copy()
        if args.max_depth is not None:
            current_xgb_params['max_depth'] = args.max_depth
            logging.info(f"max_depth 인자 오버라이드: {args.max_depth}")
        if args.eta is not None:
            current_xgb_params['eta'] = args.eta
            logging.info(f"eta 인자 오버라이드: {args.eta}")

        # ✅ feature_names 불러오기
        feature_names = load_feature_names(
            os.path.join(project_root, processed_data_dir, feature_names_file)
        )

        # ✅ 학습 실행
        train_model(
            train_filepath=os.path.join(project_root, processed_data_dir, train_data_file),
            target_column=target_column,
            feature_names=feature_names,
            xgb_params=current_xgb_params,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
        )

        logging.info("✅ 모델 학습 완료. MLflow에 결과가 기록되었습니다.")

    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)
