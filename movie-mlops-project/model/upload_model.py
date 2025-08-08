# upload_model.py

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# 💡 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅ .env 로드
load_dotenv()

from model.s3_utils import upload_to_s3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(model_filename: str):
    # 📌 환경 변수에서 버킷 정보 읽기
    bucket_name = os.getenv("S3_BUCKET_NAME")
    model_prefix = os.getenv("S3_MODEL_PREFIX")  # 예: "models/"

    if not bucket_name or not model_prefix:
        logging.error("❌ 환경 변수(S3_BUCKET_NAME, S3_MODEL_PREFIX)를 불러올 수 없습니다.")
        return

    # 📌 모델 경로
    local_model_path = os.path.join(project_root, "model_artifacts", model_filename)
    s3_key = os.path.join(model_prefix, model_filename)  # S3 내 경로

    # ✅ S3 업로드
    try:
        upload_to_s3(local_model_path, bucket_name, s3_key)
        logging.info(f"✅ 모델 S3 업로드 완료: s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"❌ 업로드 실패: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델을 S3로 업로드")
    parser.add_argument("--model_file", required=True, help="업로드할 모델 파일명 (.pkl)")
    args = parser.parse_args()
    main(args.model_file)
