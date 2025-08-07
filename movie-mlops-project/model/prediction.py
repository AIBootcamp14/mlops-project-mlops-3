import os
import sys
import logging
import argparse
import pandas as pd
import joblib

# project root 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.utils import get_config_value, load_data, load_feature_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_model(model_path: str):
    model = joblib.load(model_path)
    logging.info(f"모델 로드 완료: {model_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 XGBoost 모델로 예측을 수행합니다.")
    parser.add_argument(
        "--model_filename",
        type=str,
        help="불러올 모델 파일명 (예: xgb_md6_eta0_3.pkl)",
    )
    args = parser.parse_args()

    try:
        config_path = os.path.join(project_root, 'config', 'config.yaml')

        processed_data_dir = get_config_value(config_path, 'data.processed_data_dir')
        test_data_file = get_config_value(config_path, 'data.test_data_file')
        feature_names_file = get_config_value(config_path, 'data.feature_names_file')
        model_output_dir = os.path.join(project_root, get_config_value(config_path, 'model.output_dir'))

        # 명령줄 인자가 있으면 해당 파일 사용, 없으면 config.yaml의 기본값 사용
        if args.model_filename:
            model_filename = args.model_filename
            logging.info(f"입력된 모델 파일명 사용: {model_filename}")
        else:
            model_filename = get_config_value(config_path, 'model.filename')
            logging.info(f"기본 config 모델 파일명 사용: {model_filename}")

        model_path = os.path.join(model_output_dir, model_filename)

        # 1. 모델 로드
        model = load_trained_model(model_path)

        # 2. 테스트 데이터 로드
        test_df = load_data(os.path.join(project_root, processed_data_dir, test_data_file))
        feature_names = load_feature_names(os.path.join(project_root, processed_data_dir, feature_names_file))
        X_test = test_df[feature_names]

        # 3. 예측 수행
        predictions = model.predict(X_test)

        # 4. 결과 확인
        print("예측 결과 샘플:")
        print(predictions[:10])

    except Exception as e:
        logging.error(f"예측 중 오류 발생: {e}", exc_info=True)
