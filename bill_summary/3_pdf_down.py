# 추출한 결측값인 의안의 원안pdf 다운받음 => PDF폴더에 저장됨

import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# 설정
CSV_FILE = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/missing/bill_summary20_missing_titles.csv'
URL_COL = 'url'
SAVE_DIR = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF/pdfs'
ERROR_DIR = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF/error_html'
MAX_RETRIES = 3

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

def is_valid_file_id(file_id):
    """UUID 형식인지 간단히 검사 (36자, 하이픈 4개)"""
    return isinstance(file_id, str) and len(file_id) == 36 and file_id.count('-') == 4

def is_pdf(content):
    """파일이 PDF인지 간단히 검사"""
    return content.startswith(b'%PDF')

def extract_file_id(html):
    """HTML에서 openBillFile 호출 파라미터 중 file_id 추출"""
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.find_all('a', href=True):
        m = re.search(r"openBillFile\('([^']+)','([^']+)','([^']+)'\)", a['href'])
        if m:
            base_url, file_id, seq = m.groups()
            if is_valid_file_id(file_id):
                return file_id
    return None

def download_pdf(file_id):
    """file_id로 PDF 다운로드 시도, 최대 MAX_RETRIES회 재시도"""
    pdf_url = f"https://likms.assembly.go.kr/filegate/servlet/FileGate?type=1&bookId={file_id}"
    headers = {
        "Referer": "https://likms.assembly.go.kr/bill/billDetail.do",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(pdf_url, headers=headers, timeout=15)
            if resp.status_code == 200:
                if is_pdf(resp.content):
                    filepath = os.path.join(SAVE_DIR, f"{file_id}.pdf")
                    with open(filepath, 'wb') as f:
                        f.write(resp.content)
                    print(f"[성공] {file_id}.pdf 다운로드 완료 (시도 {attempt})")
                    return True
                else:
                    # PDF 아닌 경우 에러 HTML 저장
                    err_path = os.path.join(ERROR_DIR, f"{file_id}_error_{attempt}.html")
                    with open(err_path, 'wb') as f:
                        f.write(resp.content)
                    print(f"[실패] PDF 아님 - {file_id} (시도 {attempt}) 에러페이지 저장: {err_path}")
            else:
                print(f"[HTTP 에러] 상태코드 {resp.status_code} - {file_id} (시도 {attempt})")
        except Exception as e:
            print(f"[예외] {file_id} 다운로드 중 오류: {e} (시도 {attempt})")
        time.sleep(2)  # 재시도 전 대기
    print(f"[최종 실패] {file_id} 다운로드 실패")
    return False

def main():
    df = pd.read_csv(CSV_FILE)
    for idx, row in df.iterrows():
        url = row.get(URL_COL)
        if not isinstance(url, str) or not url.startswith('http'):
            print(f"[무시] 유효하지 않은 URL: {url}")
            continue

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"[페이지 접근 실패] {url} 상태코드: {resp.status_code}")
                continue

            file_id = extract_file_id(resp.text)
            if not file_id:
                print(f"[file_id 추출 실패] {url}")
                continue

            download_pdf(file_id)
            time.sleep(1)  # 서버 부하 방지

        except Exception as e:
            print(f"[예외] URL 처리 중 오류: {url} - {e}")

if __name__ == "__main__":
    main()
