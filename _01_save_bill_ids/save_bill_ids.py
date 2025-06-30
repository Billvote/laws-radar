# 의안 접수 목록 api에서, 의안 ID를 수집하는 코드입니다

from dotenv import load_dotenv
import requests
import os
import csv

import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# .env 파일에서 API 키 불러오기
load_dotenv()
API_KEY = os.getenv('VOTE_API')

def fetch_and_save_bill_ids(eraco: str):
    url = 'https://open.assembly.go.kr/portal/openapi/BILLRCP'

    params = {
        'KEY': API_KEY,
        'Type': 'json',
        'pIndex': 1,
        'pSize': 400,
        'ERACO': eraco
    }

    all_bill_rows = []

    # 첫 번째 요청으로 전체 건수 확인
    response = requests.get(url, params=params)
    data = response.json()

    if 'BILLRCP' not in data:
        print(f"응답 오류: 'BILLRCP' 키 없음. 응답 내용:")
        print(data)
        return

    try:
        total_count = data['BILLRCP'][0]['head'][0]['list_total_count']
    except (KeyError, IndexError) as e:
        print(f"응답 구조 오류: {e}")
        return

    total_pages = (total_count + 99) // 100
    print(f"총 {total_count}개 항목, 총 {total_pages}페이지")

    for page in range(1, total_pages + 1):
        params['pIndex'] = page
        response = requests.get(url, params=params)
        data = response.json()

        if 'BILLRCP' not in data or len(data['BILLRCP']) < 2:
            print(f"페이지 {page}에서 데이터 누락됨")
            continue

        rows = data['BILLRCP'][1].get('row', [])
        all_bill_rows.extend(rows)
        print(f"페이지 {page}: {len(rows)}개 가져옴")

    # 조건에 맞는 의안만 필터링
    valid_results = {'수정가결', '원안가결'}
    filtered_bill_ids = list({
        row['BILL_ID']
        for row in all_bill_rows
        if 'BILL_ID' in row and row.get('PROC_RSLT') in valid_results
    })

    print(f"조건에 맞는 의안 ID 수: {len(filtered_bill_ids)}")

    # # CSV 파일 저장 위치
    # OUTPUT_CSV = settings.BASE_DIR / 'save_bill_ids_01' / 'data' / f'passed_bill_ids_{eraco}.csv'


    # # CSV 저장
    # with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['bill_id'])
    #     for bill_id in filtered_bill_ids:
    #         writer.writerow([bill_id])

    # print(f"저장 완료: '{OUTPUT_CSV}'")

    return filtered_bill_ids

# 실행 예시
# fetch_and_save_bill_ids('제22대', 'update_22.csv')