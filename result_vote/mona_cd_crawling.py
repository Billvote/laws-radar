import os
import time
import csv
import requests
import random
import pandas as pd
from dotenv import load_dotenv

import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('VOTE_API')

# API 엔드포인트 및 파일 경로 설정
VOTE_API_URL = 'https://open.assembly.go.kr/portal/openapi/nojepdqqaweusdfbi'
INPUT_CSV = settings.BASE_DIR / 'save_bill_ids' / 'data' / 'passed_bill_ids_20.csv'

# bill id 목록 불러오기
def load_bill_ids(file_path: str) -> list:
    with open(file_path, encoding='utf-8') as f:
        return [row['bill_id'] for row in csv.DictReader(f)]

# 파라미터 입력: **AGE 변경 필요**
def fetch_vote_results(bill_id: str) -> list:
    """단일 의안에 대한 표결 결과 가져오기"""
    params = {
        'KEY': API_KEY,
        'Type': 'json',
        'pIndex': 1,
        'pSize': 100,
        'BILL_ID': bill_id,
        'AGE': '20'
    }
    try:
        response = requests.get(VOTE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('nojepdqqaweusdfbi', [None, {}])[1].get('row', [])
    except Exception as e:
        print(f"BILL_ID {bill_id} 처리 실패: {e}")
        return []

# mona_cd 포함해 데이터 수집
def collect_monaCd_data(bill_ids: list) -> pd.DataFrame:
    all_code = []
    for i, bill_id in enumerate(bill_ids, 1):
        print(f"[{i}/{len(bill_ids)}] BILL_ID: {bill_id}")
        all_code.extend(fetch_vote_results(bill_id))
        # time.sleep(1)
        time.sleep(random.uniform(0.2, 0.5)) # 요청 사이의 대기시간을 랜덤하게 줄이기


    if not all_code:
        print("수집된 데이터가 없습니다")
        return pd.DataFrame()

    df = pd.DataFrame(all_code)
    columns = ['AGE', 'MONA_CD', 'BILL_NO', 'BILL_ID', 'RESULT_VOTE_MOD']
    
    return df[[col for col in columns if col in df.columns]]


if __name__ == "__main__":
    bill_ids = load_bill_ids(INPUT_CSV)
    df = collect_monaCd_data(bill_ids)

# 저장
output_path = settings.BASE_DIR / 'result_vote' / 'data' / 'vote_20.csv'
df.to_csv(output_path, index=False, na_rep='NULL')
print(f"CSV 저장 완료: {output_path}")