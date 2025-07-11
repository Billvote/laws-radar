﻿# CSV 파일에서 BILL_URL 리스트와 제안이유, 주요내용 추출 하는 코드 => data폴더에 저장됨

import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import re
import time

import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 1. CSV 파일에서 BILL_URL 리스트 추출 (중복 제거)
# csv_path = settings.BASE_DIR / 'result_vote_02' / 'data' / 'vote_results_22.csv'
# try:
#     df = pd.read_csv(csv_path)
#     urls = df['BILL_URL'].dropna().unique().tolist()
# except Exception as e:
#     print(f"CSV 파일 읽기 에러: {e}")
#     urls = []

results = []

# 2. 텍스트 정제 함수 (접두사 어디에 있든 제거)
def clean_text(text):
    if not text:
        return text
    # 유니코드 공백 및 제어 문자 제거
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\r\n\t]', ' ', text)
    text = text.strip('"').strip()
    # 쉼표 제거
    text = text.replace(',', ' ')
    # 접두사 패턴 (긴 것부터, 문장 전체에서 등장하면 모두 제거)
    prefixes = [
        r'■\s*대안의\s*제안이유\s*및\s*주요내용',
        r'■\s*대안의\s*제안이유',
        r'■\s*제안이유\s*및\s*주요내용',
        r'■\s*제안이유\s*및\s*주요\s*내용',
        r'■\s*제안이유\s*,\s*주요내용',
        r'■\s*제안이유\s*,\s*주요\s*내용',
        r'■\s*제안이유\s*및',
        r'■\s*제안이유',
        r'※\s*제안이유\s*및\s*주요내용',
        r'※\s*제안이유',
        r'◇\s*제안이유\s*및\s*주요내용',
        r'◇\s*제안이유',
        r'▲\s*제안이유\s*및\s*주요내용',
        r'▲\s*제안이유',
        r'△\s*제안이유\s*및\s*주요내용',
        r'△\s*제안이유',
        r'대안의\s*제안이유\s*및\s*주요내용',
        r'대안의\s*제안이유',
        r'제안이유\s*및\s*주요\s*내용',
        r'제안이유\s*,\s*주요내용',
        r'제안이유\s*,\s*주요\s*내용',
        r'제안이유\s*및\s*주요내용',
        r'제안이유\s*및',
        r'제안\s*이유\s*및\s*주요내용',
        r'제안\s*이유',
        r'제안이유'
    ]
    prefixes = sorted(prefixes, key=len, reverse=True)
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. 크롤링 함수 (재시도 로직 포함)
def crawl(url, max_retries=3, timeout=15):
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            element = soup.select_one("div#summaryContentDiv.textType02")
            if element:
                summary_text = element.get_text(separator=' ', strip=True)
                summary_text = clean_text(summary_text)
            else:
                summary_text = pd.NA
            return {'url': url, 'content': summary_text}
        except Exception as e:
            if attempt < max_retries:
                print(f"⏳ 재시도 {attempt}/{max_retries} - {url}")
                time.sleep(2)
            else:
                print(f"에러 발생: {url}\n에러 내용: {e}")
                return {'url': url, 'content': f'ERROR: {str(e)}'}

# 4. 크롤링 및 직접 텍스트 분리 처리
def process_url(url):
    # 쉼표로 URL과 텍스트가 같이 있는 경우 분리
    if ',' in url:
        url_part, text_part = url.split(',', 1)
        url_part = url_part.strip()
        summary_text = clean_text(text_part)
        return {'url': url_part, 'content': summary_text}
    else:
        return crawl(url)
# 새로 추가-------------------------------------------
def crawl_summaries(urls, max_workers=20):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_url, url) for url in urls]
        for future in as_completed(futures):
            results.append(future.result())
    return results

# 5. CSV에서 URL 불러오기 + 크롤링 + 저장까지 한 번에 처리하는 함수
def run_crawl_and_save(csv_path, output_path):
    try:
        df = pd.read_csv(csv_path)
        urls = df['BILL_URL'].dropna().unique().tolist()
    except Exception as e:
        print(f"CSV 파일 읽기 에러: {e}")
        return
    
    results = crawl_summaries(urls)
    result_df = pd.DataFrame(results)
    merged_df = result_df.groupby('url', as_index=False)['summary'].agg(
        lambda x: ' / '.join([str(i) for i in x if pd.notna(i)])
    )
    null_count = result_df['summary'].isna().sum()
    print(f"\n요소를 찾지 못한 URL 수: {null_count}개")
    merged_df.to_csv(output_path, encoding='utf-8-sig', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"\n크롤링 및 저장 완료! 파일 위치: {output_path}")

# 기존 -----------------------------------------------
# # 5. 멀티스레드 크롤링 (최대 20개 동시 요청)
# with ThreadPoolExecutor(max_workers=20) as executor:
#     futures = [executor.submit(process_url, url) for url in urls]
#     for idx, future in enumerate(as_completed(futures), 1):
#         result = future.result()
#         print(f"[{idx}/{len(urls)}] 완료: {result['url']}")
#         results.append(result)

# # 6. 결과 DataFrame 생성
# result_df = pd.DataFrame(results)

# # 7. 중복 URL 병합 (NaN은 제외하고 합침)
# merged_df = result_df.groupby('url', as_index=False)['summary'].agg(
#     lambda x: ' / '.join([str(i) for i in x if pd.notna(i)])
# )

# # 8. NaN 개수 계산
# null_count = result_df['summary'].isna().sum()
# print(f"\n요소를 찾지 못한 URL 수: {null_count}개")

# # 9. 결과 저장 (큰따옴표 없이)
# output_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/data/bill_summary.csv'
# merged_df.to_csv(output_path, encoding='utf-8-sig', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
# print("\n크롤링 및 병합 완료! bill_summary.csv 파일이 생성되었습니다.")