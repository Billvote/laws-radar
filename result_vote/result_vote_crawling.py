import requests
import pandas as pd
import time
import csv
from dotenv import load_dotenv
from pprint import pprint
import os

# 환경 변수에서 API 키 불러오기
load_dotenv()
API_KEY = os.getenv('VOTE_API')

# 본회의 표결 결과 API 엔드포인트
VOTE_API_URL = 'https://open.assembly.go.kr/portal/openapi/nojepdqqaweusdfbi'

def load_bill_ids_from_csv(file_path: str) -> list:
    """CSV 파일에서 의안 ID 목록을 불러옵니다."""
    bill_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bill_ids.append(row['bill_id'])
    return bill_ids

def fetch_vote_results_for_bill(bill_id: str) -> list:
    """특정 의안 ID에 대한 표결 결과를 요청하여 반환합니다."""
    params = {
        'KEY': API_KEY,
        'Type': 'json',
        'pIndex': 1,
        'pSize': 100,
        'BILL_ID': bill_id,
        'AGE': '22'
    }

    try:
        response = requests.get(VOTE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        # pprint(response.json())
        data = response.json()
        return data['nojepdqqaweusdfbi'][1].get('row', [])
    except Exception as e:
        print(f"❌ 오류 발생 (BILL_ID: {bill_id}): {e}")
        return []

def collect_vote_data_as_dataframe(bill_ids: list) -> pd.DataFrame:
    """여러 의안 ID에 대해 표결 결과를 수집하고 DataFrame으로 반환합니다."""
    all_votes = []
    for i, bill_id in enumerate(bill_ids, 1):
        print(f"▶️ ({i}/{len(bill_ids)}) BILL_ID 처리 중: {bill_id}")
        vote_rows = fetch_vote_results_for_bill(bill_id)
        all_votes.extend(vote_rows)
        time.sleep(1)  # 요청 간 딜레이

    if not all_votes:
        print("⚠️ 표결 데이터가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(all_votes)

    # 실제 존재하는 컬럼만 추출
    expected_cols = ['AGE', 'HG_NM', 'POLY_NM', 'RESULT_VOTE_MOD', 'RESULT_VOTE', 'BILL_ID', 'BILL_NAME', 'VOTE_DATE']
    existing_cols = [col for col in expected_cols if col in df.columns]
    df = df[existing_cols]

    return df


# 실행 예시
if __name__ == "__main__":
    bill_ids = load_bill_ids_from_csv('passed_bill_ids_22.csv')
    df = collect_vote_data_as_dataframe(bill_ids)
    print("\n📊 최종 DataFrame 미리보기:")
    print(df.head(20))