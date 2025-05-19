import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import re
import time

# 1. CSV 파일에서 BILL_URL 리스트 추출 (중복 제거)
csv_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/result_vote/data/vote_results_20.csv'
try:
    df = pd.read_csv(csv_path)
    urls = df['BILL_URL'].dropna().unique().tolist()
except Exception as e:
    print(f"CSV 파일 읽기 에러: {e}")
    urls = []

results = []

# 2. 텍스트 정제 함수 (유니코드 특수문자까지 진단/제거)
def clean_text(text):
    if not text:
        return text
    # 모든 유니코드 공백/제어문자를 일반 공백으로 변환
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\r\n\t]', ' ', text)
    text = text.strip('"').strip()
    prefixes = sorted([
        "■ 대안의 제안이유 및 주요내용",
        "■ 대안의 제안이유",
        "■ 제안이유 및 주요내용",
        "■ 제안이유 및 주요 내용",
        "■ 제안이유, 주요내용",
        "■ 제안이유, 주요 내용",
        "■ 제안이유 및",
        "■ 제안이유",
        "※ 제안이유 및 주요내용",
        "※ 제안이유",
        "◇ 제안이유 및 주요내용",
        "◇ 제안이유",
        "▲ 제안이유 및 주요내용",
        "▲ 제안이유",
        "△ 제안이유 및 주요내용",
        "△ 제안이유",
        "대안의 제안이유 및 주요내용",
        "대안의 제안이유",
        "제안이유 및 주요내용",
        "제안이유 및 주요 내용",
        "제안이유, 주요내용",
        "제안이유, 주요 내용",
        "제안이유 및",
        "제안이유",
        "제안 이유 및 주요내용",
        "제안 이유",
    ], key=len, reverse=True)
    # 콤마 뒤에 유니코드 공백/제어문자가 있든 없든 인식
    suffix_pattern = r'(?:[\s\u200b\u200c\u200d\ufeff\xa0\r\n\t:：\-–~,，]*)(?:입니다|임|임\.|임니다|임니|이다|다|임니다)?(?:[\s\u200b\u200c\u200d\ufeff\xa0\r\n\t\.:：\-–~,，]*)(?:\n)?'
    pattern = r'(^|,[\s\u200b\u200c\u200d\ufeff\xa0\r\n\t]*)(' + '|'.join(re.escape(prefix) for prefix in prefixes) + r')' + suffix_pattern
    text = re.sub(pattern, r'\1', text, flags=re.MULTILINE)
    text = text.replace(',', ' ')
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
            return {'url': url, 'summary': summary_text}
        except Exception as e:
            if attempt < max_retries:
                print(f"⏳ 재시도 {attempt}/{max_retries} - {url}")
                time.sleep(2)
            else:
                print(f"에러 발생: {url}\n에러 내용: {e}")
                return {'url': url, 'summary': f'ERROR: {str(e)}'}

# 4. 멀티스레드 크롤링 (최대 20개 동시 요청)
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(crawl, url) for url in urls]
    for idx, future in enumerate(as_completed(futures), 1):
        result = future.result()
        print(f"[{idx}/{len(urls)}] 완료: {result['url']}")
        results.append(result)

# 5. 결과 DataFrame 생성
result_df = pd.DataFrame(results)

# 6. 중복 URL 병합 (NaN은 제외하고 합침)
merged_df = result_df.groupby('url', as_index=False)['summary'].agg(
    lambda x: ' / '.join([str(i) for i in x if pd.notna(i)])
)

# 7. NaN 개수 계산
null_count = result_df['summary'].isna().sum()
print(f"\n⚠️ 요소를 찾지 못한 URL 수: {null_count}개")

# 8. 결과 저장 (큰따옴표 없이)
output_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/data/bill_summary.csv'
merged_df.to_csv(output_path, encoding='utf-8-sig', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
print("\n✅ 크롤링 및 병합 완료! bill_summary.csv 파일이 생성되었습니다.")
