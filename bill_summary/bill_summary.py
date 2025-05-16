import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. CSV 파일에서 BILL_URL 리스트 추출 (중복 제거)
csv_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/result_vote/data/vote_results_22.csv'
try:
    df = pd.read_csv(csv_path)
    urls = df['BILL_URL'].dropna().unique().tolist()
except Exception as e:
    print(f"CSV 파일 읽기 에러: {e}")
    urls = []

results = []

# 2. 크롤링 함수 (요소 없으면 NaN으로 처리)
def crawl(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        element = soup.select_one("div#summaryContentDiv.textType02")
        
        if element:
            summary_text = element.get_text(separator=' ', strip=True)
        else:
            summary_text = pd.NA  # 요소 없으면 NaN 할당
            
        return {'url': url, 'summary': summary_text}
        
    except Exception as e:
        print(f"에러 발생: {url}\n에러 내용: {e}")
        return {'url': url, 'summary': f'ERROR: {str(e)}'}

# 3. 멀티스레드 크롤링 (최대 20개 동시 요청)
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(crawl, url) for url in urls]
    for idx, future in enumerate(as_completed(futures), 1):
        result = future.result()
        print(f"[{idx}/{len(urls)}] 완료: {result['url']}")
        results.append(result)

# 4. 결과 DataFrame 생성
result_df = pd.DataFrame(results)

# 5. 중복 URL 병합 (NaN은 제외하고 합침)
merged_df = result_df.groupby('url', as_index=False)['summary'].agg(
    lambda x: ' / '.join([str(i) for i in x if pd.notna(i)])
)

# 6. NaN 개수 계산
null_count = result_df['summary'].isna().sum()
print(f"\n⚠️ 요소를 찾지 못한 URL 수: {null_count}개")

# 7. 결과 저장
output_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/data/bill_summary.csv'
merged_df.to_csv(output_path, encoding='utf-8-sig', index=False)
print("\n✅ 크롤링 및 병합 완료! bill_summary.csv 파일이 생성되었습니다.")
