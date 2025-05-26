import os
import pandas as pd
import requests
import re
import time
import logging
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    filename='hwp_download.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 파일 경로 설정
csv_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar\bill_summary/missing/bill_summary20_missing_titles.csv'
download_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/hwp/20'
os.makedirs(download_dir, exist_ok=True)

# CSV 파일 읽기
try:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='cp949')

def extract_bill_id(url):
    """URL에서 billId 추출"""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    return params.get('billId', [None])[0]

def download_hwp(session, url, bill_id):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Referer": url,
            "Origin": "http://likms.assembly.go.kr"
        }

        # 1. 의안 상세 페이지 요청
        response = session.get(url, headers=headers)
        response.raise_for_status()

        # 2. JavaScript 함수 파싱
        pattern = r"openBillFile\('([^']+)','([^']+)','([^']+)'\)"
        match = re.search(pattern, response.text)
        if not match:
            raise ValueError("openBillFile 패턴 미발견")

        base_url, atch_file_id, file_type = match.groups()
        download_url = f"{base_url}?bookId={atch_file_id}&type={file_type}"

        # 3. 파일 다운로드 요청
        file_response = session.post(
            download_url,
            headers=headers,
            data={"bookId": atch_file_id, "type": file_type},
            stream=True
        )
        file_response.raise_for_status()

        # 4. 파일 유효성 검증 (HWP 시그니처)
        if file_response.content[:8] != b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':
            raise ValueError("잘못된 HWP 파일 형식")

        # 5. 파일 저장
        filename = os.path.join(download_dir, f"{bill_id}.hwp")
        with open(filename, 'wb') as f:
            for chunk in file_response.iter_content(8192):
                f.write(chunk)
        
        return True

    except Exception as e:
        logging.error(f"[{bill_id}] 오류: {str(e)}")
        return False

def process_batch():
    success_count = 0
    total = len(df)
    
    with requests.Session() as session:
        session.headers.update({
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "DNT": "1"
        })

        for idx, row in tqdm(df.iterrows(), total=total, desc="진행상황"):
            url = row['url']
            bill_id = extract_bill_id(url)
            
            if not bill_id:
                logging.error(f"잘못된 URL: {url}")
                continue

            try:
                if download_hwp(session, url, bill_id):
                    success_count += 1
                time.sleep(1.5)  # 서버 부하 방지
            except Exception as e:
                logging.critical(f"치명적 오류 ({idx}행): {str(e)}")
    
    print(f"▣ 최종 결과: {success_count}/{total} 성공 ({success_count/total*100:.2f}%)")

if __name__ == "__main__":
    process_batch()
