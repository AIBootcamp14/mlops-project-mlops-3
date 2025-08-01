import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data(filepath: str) -> pd.DataFrame:
    """
    지정된 경로에서 CSV 파일을 로드합니다.

    Args:
        filepath (str): 로드할 CSV 파일의 경로.

    Returns:
        pd.DataFrame: 로드된 데이터프레임.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때 발생합니다.
        Exception: 데이터 로드 중 알 수 없는 오류가 발생할 때 발생합니다.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"데이터 로드 성공: {filepath}, Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {filepath}")
        raise
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {e}")
        raise


def save_model(model, filepath: str):
    """
    학습된 모델을 지정된 경로에 저장합니다.

    Args:
        model: 저장할 모델 객체 (예: scikit-learn, XGBoost 모델).
        filepath (str): 모델을 저장할 파일 경로.

    Raises:
        Exception: 모델 저장 중 오류가 발생할 때 발생합니다.
    """
    try:
        joblib.dump(model, filepath)
        logging.info(f"모델 저장 성공: {filepath}")
    except Exception as e:
        logging.error(f"모델 저장 중 오류 발생: {e}")
        raise


def load_model(filepath: str):
    """
    지정된 경로에서 모델을 로드합니다.

    Args:
        filepath (str): 로드할 모델 파일의 경로.

    Returns:
        Any: 로드된 모델 객체.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때 발생합니다.
        Exception: 모델 로드 중 알 수 없는 오류가 발생할 때 발생합니다.
    """
    try:
        model = joblib.load(filepath)
        logging.info(f"모델 로드 성공: {filepath}")
        return model
    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {filepath}")
        raise
    except Exception as e:
        logging.error(f"모델 로드 중 오류 발생: {e}")
        raise


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    모델의 성능을 평가하고 MSE, RMSE, R2, MAE 지표를 반환합니다.

    Args:
        y_true (np.ndarray): 실제 타겟 값 배열.
        y_pred (np.ndarray): 모델이 예측한 타겟 값 배열.

    Returns:
        dict: MSE, RMSE, R2, MAE를 포함하는 딕셔너리.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
    }
    logging.info(f"모델 평가 결과: {metrics}")
    return metrics


def load_feature_names(filepath: str) -> list:
    """
    JSON 파일에서 특성 이름을 로드합니다.

    Args:
        filepath (str): 특성 이름이 저장된 JSON 파일의 경로.

    Returns:
        list: 로드된 특성 이름 문자열 리스트.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때 발생합니다.
        Exception: 특성 이름 로드 중 알 수 없는 오류가 발생할 때 발생합니다.
    """
    try:
        with open(filepath, 'r') as f:
            feature_names = json.load(f)

        logging.info(
            f"특성 이름 로드 성공: {filepath}, 특성 수: {len(feature_names)}"
        )
        return feature_names
    except FileNotFoundError:
        logging.error(f"특성 이름 파일을 찾을 수 없습니다: {filepath}")
        raise
    except Exception as e:
        logging.error(f"특성 이름 로드 중 오류 발생: {e}")
        raise


def get_config_value(config_path: str, key_path: str):
    """
    YAML 설정 파일에서 지정된 키 경로의 값을 가져옵니다.

    Args:
        config_path (str): YAML 설정 파일의 경로.
        key_path (str): 점(.)으로 구분된 키 경로 (예: 'data.processed_data_dir').

    Returns:
        Any: 지정된 키 경로에 해당하는 값.

    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않을 때 발생합니다.
        KeyError: 지정된 키 경로를 설정 파일에서 찾을 수 없을 때 발생합니다.
        Exception: 설정 값 가져오기 중 알 수 없는 오류가 발생할 때 발생합니다.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)  # 누락된 부분 추가

        keys = key_path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(
                    f"설정 파일에 '{key_path}' 키가 없습니다. '{key}'를 찾을 수 없습니다."
                )
        return value
    except FileNotFoundError:
        logging.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise
    except KeyError as ke:
        logging.error(f"설정 값 가져오기 중 오류 발생: {ke}")
        raise
    except Exception as e:
        logging.error(f"설정 값 가져오기 중 알 수 없는 오류 발생: {e}")
        raise