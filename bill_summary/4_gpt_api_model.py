# gqt api이용해서 pdf에서 의안번호, 제안이유, 주요내용 csv로 추출

import os
import sys
from pathlib import Path
import pandas as pd
import time
import pdfplumber
from openai import OpenAI
import re
from dotenv import load_dotenv  # 추가

# 환경 변수 로드
load_dotenv()  # 추가

# settings.py가 있는 폴더(laws-radar)를 sys.path에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))
import settings

# PDF 폴더와 결과 폴더 경로 (settings.BASE_DIR 기준 상대경로)
pdf_folder = settings.BASE_DIR / 'bill_summary' / 'PDF' / '21'
output_dir = settings.BASE_DIR / 'bill_summary' / 'PDF_summary'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "output21.csv"

pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

# Perplexity API 설정 (.env에서 키 가져오기)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")  # 수정
if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY가 .env 파일에 설정되지 않았습니다.")  # 추가

client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

def pdf_to_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_info(text):
    prompt = f"""
    PDF 텍스트에서 다음 정보를 정확히 추출:
    1. 의안번호: 문서 내 표기된 의안번호(숫자만, 자리수 제한 없음)
    2. 제안이유: "제안이유" 섹션 전체 텍스트 (없으면 빈값)
    3. 주요내용: "주요내용" 또는 "주 문" 섹션 전체 텍스트 (없으면 빈값, 두 항목 중 하나라도 있으면 추출)
    
    반드시 아래와 같은 JSON 형식만 출력:
    {{"bill_no": "숫자만", "reason": "...", "main": "..."}}
    
    원문:
    {text[:6000]}
    """
    try:
        response = client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        import json
        content = response.choices[0].message.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        json_str = content[json_start:json_end]
        result = json.loads(json_str)
        bill_no = ''.join(re.findall(r'\d+', str(result.get("bill_no", ""))))
        return {
            "bill_no": bill_no,
            "reason": result.get("reason", "").replace('\n', ' ').replace('\r', ' ').strip(),
            "main": result.get("main", "").replace('\n', ' ').replace('\r', ' ').strip()
        }
    except Exception as e:
        print(f"⚠️ API 오류 또는 파싱 실패: {e}")
        return {"bill_no": "", "reason": "", "main": ""}

results = []
for file in pdf_files:
    file_path = pdf_folder / file
    text = pdf_to_text(file_path)
    data = extract_info(text)
    if not data['bill_no'] and not data['reason'] and not data['main']:
        continue
    row = [file, data['bill_no'], data['reason'], data['main']]
    results.append(row)
    time.sleep(1.5)

df = pd.DataFrame(results, columns=["파일명", "의안번호", "제안이유", "주요내용or주 문"])
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n✅ {len(results)}개 파일 처리 완료! 결과: {output_path}")
