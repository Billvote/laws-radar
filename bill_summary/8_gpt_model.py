import os
import pdfplumber
import pandas as pd
import re
import json
import time
import requests

# 1. Perplexity API 키 입력
PERPLEXITY_API_KEY = "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"  # 예: "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 2. PDF 폴더 경로
pdf_folder = "C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF/22"  # 상대경로 지정하기
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

def extract_sections_with_perplexity(text):
    try:
        # 입력 텍스트 전처리
        text = text[:6000].replace('\n', ' ').replace('"', "'")
        
        # 강화된 프롬프트
        prompt = (
            '아래 텍스트에서 "의안번호", "제안이유", "주요내용" **만** JSON 형식으로 반환하세요.\n'
            '반드시 {"의안번호": "...", "제안이유": "...", "주요내용": "..."} 형식으로 출력해주세요.\n'
            '추가 설명이나 텍스트는 절대 포함하지 마세요.\n'
            '---\n'
            f'{text}'
        )
        
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [
            {
                "role": "system",
                "content": "반드시 요청한 3개 필드만 포함한 raw JSON을 출력해야 합니다. 다른 텍스트는 허용되지 않습니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        payload = {
            "model": "sonar",  # 최신 지원 모델명
            "messages": messages,
            "temperature": 0
        }
        
        # API 호출
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # HTTP 오류 처리
        if response.status_code != 200:
            print(f"\n⚠️ API 오류 발생 [HTTP {response.status_code}]")
            print(f"응답 내용: {response.text}")
            return {"의안번호": "", "제안이유": "", "주요내용": ""}
            
        # 응답 구조 검증
        response_data = response.json()
        if ('choices' not in response_data) or (len(response_data['choices']) == 0):
            print("\n⚠️ 잘못된 응답 구조")
            print(f"전체 응답: {response_data}")
            return {"의안번호": "", "제안이유": "", "주요내용": ""}
            
        # 컨텐츠 추출
        content = response_data['choices'][0]['message']['content']
        if not content.strip():
            print("\n⚠️ 빈 응답 수신")
            return {"의안번호": "", "제안이유": "", "주요내용": ""}
            
        # JSON 추출 강화 (코드블록 및 "json" 접두사 제거)
        json_str = content  # 기본값 할당
        # 코드블록이 있으면 제거
        codeblock_match = re.search(r"``````", content, re.IGNORECASE)
        if codeblock_match:
            json_str = codeblock_match.group(1).strip()
        # "json" 접두사 제거
        json_str = re.sub(r'^json', '', json_str, flags=re.IGNORECASE).strip()
        
        # JSON 파싱
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON 파싱 실패: {str(e)}")
            print(f"원본 컨텐츠: {json_str}")
            return {"의안번호": "", "제안이유": "", "주요내용": ""}
            
        # 필드 필터링
        return {
            "의안번호": data.get("의안번호", ""),
            "제안이유": data.get("제안이유", "").replace('\n', ' '),
            "주요내용": data.get("주요내용", "").replace('\n', ' ')
        }
        
    except Exception as e:
        print(f"\n⚠️ 치명적 오류: {str(e)}")
        return {"의안번호": "", "제안이유": "", "주요내용": ""}

def pdf_to_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

results = []

for idx, file in enumerate(pdf_files, 1):
    try:
        file_path = os.path.join(pdf_folder, file)
        print(f"\n[{idx}/{len(pdf_files)}] 처리 중: {file}")
        
        text = pdf_to_text(file_path)
        data = extract_sections_with_perplexity(text)
        
        results.append({
            "파일명": file,
            "의안번호": data["의안번호"],
            "제안이유": data["제안이유"],
            "주요내용": data["주요내용"]
        })
        
        time.sleep(1.5)  # Rate limit 대응
        
    except Exception as e:
        print(f"\n⚠️ 파일 처리 실패: {file}")
        print(f"오류: {str(e)}")
        continue

df = pd.DataFrame(results)
df.to_csv("output.csv", index=False, encoding="utf-8-sig")
print("\n✅ 처리 완료! output.csv 파일을 확인하세요.")
