import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(
    filename='hwp_download.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 파일 경로 설정
csv_path = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/missing/bill_summary22_missing_titles.csv'
download_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/downloads'
os.makedirs(download_dir, exist_ok=True)

# CSV 파일 읽기 (UTF-8 BOM 대응)
try:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='cp949')

def extract_bill_id(url):
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    return params.get('billId', [None])[0]

def download_hwp(row):
    try:
        url = row['url']
        bill_id = extract_bill_id(url)
        
        if not bill_id:
            raise ValueError(f"Invalid URL: {url}")

        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Referer": url,
            "Origin": "http://likms.assembly.go.kr"
        }

        # 1. 의안 상세 페이지 요청
        response = session.get(url, headers=headers)
        response.raise_for_status()

        # 2. JavaScript 함수 파싱 (새로운 패턴)
        pattern = r"openBillFile\('([^']+)','([^']+)','([^']+)'\)"
        match = re.search(pattern, response.text)
        
        if not match:
            raise ValueError("openBillFile pattern not found")

        base_url, atch_file_id, file_type = match.groups()
        download_url = f"{base_url}?bookId={atch_file_id}&type={file_type}"

        # 3. 실제 파일 다운로드 요청 (POST 방식)
        file_response = session.post(
            download_url,
            headers=headers,
            data={"bookId": atch_file_id, "type": file_type},
            stream=True
        )
        file_response.raise_for_status()

        # 4. 파일 유효성 검증 (HWP 매직 넘버 확인)
        if file_response.content[:8] != b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1':
            raise ValueError("Invalid HWP file signature")

        # 5. 파일 저장
        filename = os.path.join(download_dir, f"{bill_id}.hwp")
        with open(filename, 'wb') as f:
            for chunk in file_response.iter_content(8192):
                f.write(chunk)
        
        return True

    except Exception as e:
        logging.error(f"[{bill_id}] Error: {str(e)}")
        return False

# 강화된 배치 처리 시스템
def process_batch():
    success_count = 0
    total = len(df)
    
    # 세션 유지를 위한 전역 세션
    global_session = requests.Session()
    global_session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "DNT": "1"
    })

    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing"):
        try:
            if download_hwp(row):
                success_count += 1
        except Exception as e:
            logging.critical(f"Critical error at row {idx}: {str(e)}")
        finally:
            # 서버 부하 방지 딜레이
            time.sleep(1.5)
    
    print(f"최종 결과: {success_count}/{total} 성공 ({success_count/total*100:.2f}%)")

if __name__ == "__main__":
    process_batch()
