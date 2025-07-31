import logging
import os
import sys
import argparse

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

# model.utils 임포트 시 경로 문제 해결을 위해 project_root를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.utils import get_config_value, load_data, load_feature_names, save_model

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def train_model(
    train_filepath: str,
    target_column: str,
    feature_names: list,
    xgb_params: dict,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    model_output_dir: str,
    model_filename: str
) -> tuple:
    """
    XGBoost 모델을 학습하고 MLflow에 파라미터를 로깅합니다.
    학습된 모델 객체와 MLflow Run ID를 반환합니다.
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
            
            # feature_names는 이미 함수 인자로 받으므로, feature_names_filepath를 구성할 필요 없음
            # 그러나 load_feature_names 함수 자체는 파일 경로를 받으므로, 
            # 호출부에서 feature_names_filepath를 통해 로드하여 인자로 전달해야 함
            # 현재 코드에서 feature_names가 리스트로 넘어온다고 가정하고 진행
            
            X_train = train_df[feature_names]
            y_train = train_df[target_column]

            logging.info(
                f"학습 데이터 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
            )

            mlflow.log_params(xgb_params)
            logging.info(f"MLflow에 파라미터 로깅 완료: {xgb_params}")

            model = xgb.XGBRegressor(**xgb_params)
            logging.info(f"XGBoost 모델 학습 시작 (파라미터: {xgb_params})")
            model.fit(X_train, y_train)
            logging.info("XGBoost 모델 학습 완료")
            
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name=mlflow_experiment_name,
            )
            logging.info("MLflow에 모델 아티팩트 로깅 완료")

            model_full_path = os.path.join(model_output_dir, model_filename)
            os.makedirs(model_output_dir, exist_ok=True)
            save_model(model, model_full_path)
            logging.info(f"모델 로컬 저장 완료: {model_full_path}")

            return model, run_id

        except Exception as e:
            logging.error(f"모델 학습 중 치명적인 오류 발생: {e}", exc_info=True)
            mlflow.end_run(status="FAILED")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost 모델 학습 스크립트")
    parser.add_argument("--max_depth", type=int, help="XGBoost max_depth 파라미터 (기본값을 오버라이드합니다)")
    parser.add_argument("--eta", type=float, help="XGBoost eta (learning_rate) 파라미터 (기본값을 오버라이드합니다)")
    args = parser.parse_args()

    try:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml'
        )

        processed_data_dir = get_config_value(config_path, 'data.processed_data_dir')
        train_data_file = get_config_value(config_path, 'data.train_data_file')
        feature_names_file = get_config_value(config_path, 'data.feature_names_file')
        target_column = get_config_value(config_path, 'model.target_column')
        
        base_xgb_params = get_config_value(config_path, 'model.xgboost_params')
        
        mlflow_tracking_uri = get_config_value(config_path, 'mlflow.tracking_uri')
        mlflow_experiment_name = get_config_value(config_path, 'mlflow.experiment_name')
        
        model_output_dir = os.path.join(project_root, get_config_value(config_path, 'model.output_dir'))
        model_filename = get_config_value(config_path, 'model.filename')

        current_xgb_params = base_xgb_params.copy() 
        
        if args.max_depth is not None:
            current_xgb_params['max_depth'] = args.max_depth
            logging.info(f"명령줄 인자로 max_depth가 {args.max_depth}로 오버라이드되었습니다.")
        
        if args.eta is not None:
            current_xgb_params['eta'] = args.eta
            logging.info(f"명령줄 인자로 eta가 {args.eta}로 오버라이드되었습니다.")
            
        trained_model, run_id = train_model(
            train_filepath=os.path.join(project_root, processed_data_dir, train_data_file),
            target_column=target_column,
            feature_names=load_feature_names(os.path.join(project_root, processed_data_dir, feature_names_file)),
            xgb_params=current_xgb_params,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            model_output_dir=model_output_dir,
            model_filename=model_filename
        )
        
        logging.info(f"모델 학습이 완료되었습니다. MLflow Run ID: {run_id}")
        
        temp_run_id_path = os.path.join(project_root, '.temp_mlflow_run_id.txt')
        with open(temp_run_id_path, 'w') as f:
            f.write(run_id)
        logging.info(f"MLflow Run ID를 임시 파일에 저장했습니다: {temp_run_id_path}")

    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)