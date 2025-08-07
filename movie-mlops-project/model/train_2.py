import logging
import os
import sys
import argparse

import pandas as pd
import xgboost as xgb

# model.utils 임포트 경로 설정
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
    model_output_dir: str,
    model_filename: str
):
    """
    XGBoost 모델을 학습하고 로컬에 .pkl 파일로 저장합니다.
    """
    logging.info("모델 학습을 시작합니다.")

    try:
        train_df = load_data(train_filepath)
        X_train = train_df[feature_names]
        y_train = train_df[target_column]

        logging.info(
            f"학습 데이터 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
        )

        model = xgb.XGBRegressor(**xgb_params)
        logging.info(f"XGBoost 모델 학습 시작 (파라미터: {xgb_params})")
        model.fit(X_train, y_train)
        logging.info("XGBoost 모델 학습 완료")

        model_full_path = os.path.join(model_output_dir, model_filename)
        os.makedirs(model_output_dir, exist_ok=True)
        save_model(model, model_full_path)
        logging.info(f"모델 로컬 저장 완료: {model_full_path}")

        return model

    except Exception as e:
        logging.error(f"모델 학습 중 오류 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost 모델 학습 스크립트 (MLflow 미사용)")
    parser.add_argument("--max_depth", type=int, help="XGBoost max_depth 파라미터")
    parser.add_argument("--eta", type=float, help="XGBoost eta (learning_rate) 파라미터")
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
        model_output_dir = os.path.join(project_root, get_config_value(config_path, 'model.output_dir'))
        model_filename = get_config_value(config_path, 'model.filename')

        current_xgb_params = base_xgb_params.copy()

        if args.max_depth is not None:
            current_xgb_params['max_depth'] = args.max_depth
            logging.info(f"max_depth 오버라이드됨: {args.max_depth}")
        if args.eta is not None:
            current_xgb_params['eta'] = args.eta
            logging.info(f"eta 오버라이드됨: {args.eta}")

        # 하이퍼파라미터 기반 파일명 생성
        if args.max_depth is not None and args.eta is not None:
            eta_str = str(args.eta).replace('.', '_')  # 파일명에서 점(.) 제거
            model_filename = f"xgb_md{args.max_depth}_eta{eta_str}.pkl"
        else:
            model_filename = get_config_value(config_path, 'model.filename')
            logging.info(f"기본 파일명 사용: {model_filename}")

        model = train_model(
            train_filepath=os.path.join(project_root, processed_data_dir, train_data_file),
            target_column=target_column,
            feature_names=load_feature_names(os.path.join(project_root, processed_data_dir, feature_names_file)),
            xgb_params=current_xgb_params,
            model_output_dir=model_output_dir,
            model_filename=model_filename
        )

        logging.info("모델 학습 및 저장이 완료되었습니다.")

    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)
