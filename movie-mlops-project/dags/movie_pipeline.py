import subprocess
import datetime
import os
import sys

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"pipeline_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def run_step(step_name, command):
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n[STEP] {step_name} 시작 - {datetime.datetime.now()}\n")
        log.flush()
        try:
            subprocess.run(command, check=True, stdout=log, stderr=log, text=True)
            log.write(f"[STEP] {step_name} 완료 ✅\n")
        except subprocess.CalledProcessError as e:
            log.write(f"[ERROR] {step_name} 실패 ❌\n{str(e)}\n")
            print(f"❌ {step_name} 단계에서 오류 발생. 로그를 확인하세요.")
            sys.exit(1)

def main():
    print("🚀 자동화 파이프라인 시작")

    # 1. 크롤링 (crawler.py가 standalone으로 실행 가능해야 함)
    run_step("크롤링", ["python", "crawler.py"])

    # 2. 전처리 (preprocessing.py를 모듈로 import하지 않고 별도 실행한다면 여기에 추가)
    run_step("전처리", ["python", "preprocessing.py"])

    # 3. main.py 실행 (전체 파이프라인 또는 추가 작업)
    run_step("main 실행", ["python", "main.py"])

    print("✅ 자동화 파이프라인 완료!")
    print(f"📜 로그 위치: {log_file}")

if __name__ == "__main__":
    main()
