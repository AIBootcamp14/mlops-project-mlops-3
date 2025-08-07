# model/s3_utils.py
import boto3
import logging
import os

def upload_to_s3(local_path: str, bucket_name: str, s3_key: str):
    s3 = boto3.client("s3")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"파일 없음: {local_path}")
    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        logging.info(f"✅ 업로드 성공: s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"S3 업로드 실패: {e}")
        raise

def download_from_s3(bucket_name: str, s3_key: str, local_path: str):
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info(f"✅ 다운로드 성공: {local_path}")
    except Exception as e:
        logging.error(f"S3 다운로드 실패: {e}")
        raise
