# 의안내용 한줄요약 gmini 개조식 버전

import pandas as pd
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

def initialize_system():
    """Gemini API 클라이언트 초기화"""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        client = genai.Client(api_key=api_key)
        print("✅ Gemini API 초기화 완료")
        return client
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {str(e)}")
        return None

def generate_summary(client, original_text, max_retries=5):
    """
    법안 요약 특화 프롬프트와 후처리 적용
    - 3~5개 핵심 키워드/구절을 '·'로 구분
    - 각 구절은 10자 내외 명사형, 쉼표 없이 작성
    - 예시 스타일로 출력
    """
    if pd.isna(original_text) or not original_text.strip():
        return original_text
    
    for attempt in range(max_retries):
        try:
            prompt = f"""
[법안 요약 규칙]
1. 핵심 내용 3~5개를 '·' 기호로 구분해 나열
2. 각 항목은 10자 내외의 명사형 구절로 표현
3. 반드시 포함해야 할 요소:
   - 법 개정 목적(예: 실효성 제고, 처벌 강화)
   - 주요 변경 사항(예: 근거 신설, 절차 개선)
   - 관련 기관(예: 국회, 감사원)
4. 쉼표 대신 '·'와 '및' 사용
5. 150자 이내 완결성

[원문]
{original_text}

[예시 출력]
국회 증인 출석요구 실효성 제고 · 모욕죄·불출석죄 처벌 강화 · 개인정보 제공 근거 신설 등 증언제도 개선

[요약]
"""
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            summary = response.text.strip()
            # 후처리: 쉼표 → '·', ' 및 ' → '·', 연속 공백 제거, 길이 제한
            summary = summary.replace(",", "·").replace(" 및 ", "·")
            summary = ' '.join(summary.split())
            if len(summary) > 150:
                summary = summary[:147] + "..."
            return summary
        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "UNAVAILABLE" in err_msg:
                wait = random.randint(30, 60) * (attempt + 1)
                print(f"⚠️ 503 오류: {wait}초 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "429" in err_msg:
                print(f"⏳ 429 오류: 60초 대기 후 재시도")
                time.sleep(60)
            else:
                print(f"⚠️ 요약 실패: {err_msg}")
                return original_text
    return original_text

def process_csv_file(client, input_path, output_path):
    """병렬 처리로 속도 향상"""
    try:
        df = pd.read_csv(input_path, engine='python')
        print(f"📂 파일 로드 완료: {len(df)}개 행")
        
        if 'content' not in df.columns:
            raise KeyError("'content' 컬럼이 존재하지 않습니다.")
            
        total_rows = len(df)
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for idx in range(total_rows):
                original = df.at[idx, 'content']
                future = executor.submit(generate_summary, client, original)
                futures.append( (idx, future) )
                if (idx+1) % 2 == 0:
                    elapsed = time.time() - start_time
                    target_time = 8.0
                    sleep_time = max(target_time - elapsed, 0)
                    time.sleep(sleep_time)
                    start_time = time.time()
            for idx, future in futures:
                df.at[idx, 'content'] = future.result()
                if (idx+1) % 10 == 0:
                    print(f"진행률: {idx+1}/{total_rows} ({((idx+1)/total_rows)*100:.1f}%)")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장 완료: {output_path}")
        print("\n=== 처리 결과 미리보기 ===")
        for i in range(min(3, len(df))):
            print(f"[요약 {i+1}] {df.iloc[i]['content'][:100]}...")
    except Exception as e:
        print(f"❌ CSV 처리 오류: {str(e)}")

if __name__ == "__main__":
    INPUT_PATH = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\bill_filtered_final.csv"
    OUTPUT_PATH = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\processed_bills2_optimized.csv"
    
    gemini_client = initialize_system()
    if gemini_client:
        process_csv_file(gemini_client, INPUT_PATH, OUTPUT_PATH)
