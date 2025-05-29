# 반드시 설정 변경해야 할 것!!
    # 15줄: INPUT_CSV, OUTPUT_CSV 설정 변경
    # 25줄: pararms의 AGE 변경
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

INPUT_CSV = settings.BASE_DIR / 'save_bill_ids' / 'data' / 'passed_bill_ids_22.csv'
OUTPUT_CSV = settings.BASE_DIR / 'result_vote' / 'data' / 'final_22.csv'

def load_bill_ids(file_path: str) -> list:
    """CSV 파일에서 bill_id 목록 불러오기"""
    with open(file_path, encoding='utf-8') as f:
        return [row['bill_id'] for row in csv.DictReader(f)]

# AGE 변경 필수!!!!!
def fetch_vote_results(bill_id: str) -> list:
    """단일 의안에 대한 표결 결과 가져오기"""
    params = {
        'KEY': API_KEY,
        'Type': 'json',
        'pIndex': 1,
        'pSize': 400,
        'BILL_ID': bill_id,
        'AGE': '22'
    }
    try:
        response = requests.get(VOTE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('nojepdqqaweusdfbi', [None, {}])[1].get('row', [])
    except Exception as e:
        print(f"❌ BILL_ID {bill_id} 처리 실패: {e}")
        return []

def collect_vote_data(bill_ids: list) -> pd.DataFrame:
    """여러 의안 ID에 대해 표결 결과 수집"""
    all_votes = []
    for i, bill_id in enumerate(bill_ids, 1):
        print(f"[{i}/{len(bill_ids)}] BILL_ID: {bill_id}")
        all_votes.extend(fetch_vote_results(bill_id))
        time.sleep(random.uniform(0.2, 0.5)) # 요청 사이의 대기시간을 랜덤하게 줄이기

    if not all_votes:
        print("⚠️ 수집된 표결 데이터가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(all_votes)
    columns = ['AGE', 'MONA_CD', 'BILL_ID', 'BILL_NO', 'RESULT_VOTE_MOD', 'VOTE_DATE']
    
    return df[[col for col in columns if col in df.columns]]

def save_to_csv(df: pd.DataFrame, filename: str):
    """DataFrame을 CSV로 저장"""
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n CSV 저장 완료: {filename}")

if __name__ == "__main__":
    bill_ids = load_bill_ids(INPUT_CSV)
    df = collect_vote_data(bill_ids)
    if not df.empty:
        save_to_csv(df, OUTPUT_CSV)