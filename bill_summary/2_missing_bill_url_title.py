# 결측값인 의안 url, title 추출됨 => missing폴더에 저장됨

import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 1. 파일 읽기 & 결측값 추출
file_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/data/bill_summary20.csv'
try:
    df = pd.read_csv(file_path)
    # 결측값이 있는 행만 추출 (모든 컬럼 기준)
    df_na = df[df.isnull().any(axis=1)]
    # 결측값 행의 url만 추출, 중복 제거
    missing_urls = df_na['url'].dropna().unique().tolist()
except FileNotFoundError:
    print("⚠️ 파일 경로를 확인해 주세요!")
    exit()

# 2. 크롤링 설정
session = requests.Session()
headers = {'User-Agent': 'Mozilla/5.0'}

def get_title(url):
    """URL에서 의안명 추출"""
    try:
        response = session.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h3', class_='bill_title') or soup.find('h3')
        return title_tag.get_text(strip=True) if title_tag else '의안명 없음'
    except Exception as e:
        return f'오류: {str(e)}'

# 3. 병렬 처리로 타이틀 수집
results = []
if missing_urls:
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_title, url): url for url in missing_urls}
        for idx, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            title = future.result()
            results.append({'url': url, 'bill_title': title})
            print(f"[{idx}/{len(missing_urls)}] {url} → {title}")
else:
    print("결측값 URL이 없습니다.")

# 4. 결과 저장 (지정 폴더로)
output_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/missing'
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성

output_path = os.path.join(output_dir, 'bill_summary20_missing_titles.csv')
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\n✅ 완료! 저장 위치: {output_path}")
