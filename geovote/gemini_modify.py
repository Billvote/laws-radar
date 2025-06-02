import pandas as pd
import os
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

# ================ 시스템 초기화 ================
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

# ================ 관형격 '의' 자동 삽입 ================
def insert_possessive_eui(phrase):
    """
    명사+명사 구조에서 자연스럽게 '의'를 넣어줌
    예: '고엽제 환자 생계 어려움' → '고엽제 환자의 생계 어려움'
    """
    # 이미 '의'가 있으면 그대로 반환
    if '의' in phrase:
        return phrase
    # '명사 명사' 패턴만 추출
    tokens = phrase.split()
    # 3개 이상 명사라면 마지막 두 단어 사이에 '의' 삽입
    if len(tokens) >= 3 and all(re.match(r'^[가-힣]+$', t) for t in tokens[-3:]):
        tokens.insert(-2, '의')
        return ' '.join(tokens)
    # 2개 명사라면 마지막 두 단어 사이에 '의' 삽입
    elif len(tokens) >= 2 and all(re.match(r'^[가-힣]+$', t) for t in tokens[-2:]):
        tokens.insert(-1, '의')
        return ' '.join(tokens)
    return phrase

# ================ 자연스러운 조사/어미 변환 ================
def josa(word, josa_pair):
    """받침 유무에 따라 적합한 조사를 반환"""
    if not word:
        return josa_pair[1]
    code = ord(word[-1])
    has_jong = (code - 44032) % 28 != 0
    return josa_pair[0] if has_jong else josa_pair[1]

def postprocess_korean_sentence(p1, p2, p3):
    """
    맥락과 문법을 고려한 조사 및 어미 자동 적용 + 관형격 '의' 자동 삽입
    """
    # 명사+명사 구조라면 '의' 넣기
    p1 = insert_possessive_eui(p1)
    # 목적/활성화 등은 '을/를 위해'
    if any(x in p1 for x in ["목적", "활성화", "확대", "제고", "강화", "달성"]):
        p1_clean = re.sub(r'^(목적:|목적|활성화:)', '', p1).strip()
        조사1 = josa(p1_clean, ("을", "를"))
        조사2 = josa(p2, ("을", "를"))
        조사3 = josa(p3, ("을", "를"))
        return f"{p1_clean}{조사1} 위해 {p2}{조사2} 추진하여 {p3}{조사3} 달성하고자 합니다"
    # 문제 중심: '문제는 ...에서 비롯되어 ...로 개선'
    else:
        p1_clean = re.sub(r'^(문제:|문제)', '', p1).strip()
        조사1 = josa(p1_clean, ("은", "는"))
        조사2 = josa(p2, ("에서", "에서"))
        조사3 = josa(p3, ("으로", "로"))
        return f"{p1_clean}{조사1} {p2}{조사2} 비롯되어 {p3}{조사3} 개선하고자 합니다"

def convert_to_sentence(summary):
    """3단 구조 요약을 자연스러운 한 문장으로 변환"""
    parts = [p.split(": ")[1].strip() if ": " in p else p.strip()
             for p in summary.split("→")]
    if len(parts) != 3:
        return summary.replace(",", " 그리고")
    p1, p2, p3 = parts
    return postprocess_korean_sentence(p1, p2, p3)

# ================ AI 요약 엔진 ================
def generate_summary(client, original_text, max_retries=5):
    """3단계 구조화 요약 생성 후 자연스러운 한 문장으로 변환"""
    if pd.isna(original_text) or not original_text.strip():
        return original_text

    for attempt in range(max_retries):
        try:
            prompt = f"""
[법안 요약 규칙]
1. 다음 형식 중 가장 적합한 구조 선택:
   - 문제: [핵심 문제] → 원인: [주요 원인] → 해결: [제안된 해결책]
   - 배경: [발생 배경] → 내용: [주요 조항] → 효과: [기대 효과]
   - 목적: [개정 목적] → 방법: [시행 방법] → 결과: [예상 결과]
2. 각 부분은 15~20자 내외로 간결하게 작성
3. 쉼표 대신 '→' 기호 사용
4. 반드시 3개 요소 포함

[원문]
{original_text}

[예시 출력]
문제: 증인 소환 권한 미흡 → 원인: 불출석 처벌 근거 부재 → 해결: 처벌 조항 신설

[요약]
"""
            response = client.models.generate_content(
                model='gemini-1.5-flash-latest',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=150
                )
            )
            summary = response.text.strip()
            return convert_to_sentence(summary)
        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "UNAVAILABLE" in err_msg:
                wait = random.randint(10, 30) * (attempt + 1)
                print(f"⚠️ 503 오류: {wait}초 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "429" in err_msg:
                print(f"⏳ 429 오류: 30초 대기 후 재시도")
                time.sleep(30)
            else:
                print(f"⚠️ 요약 실패: {err_msg}")
                return original_text
    return original_text

# ================ 병렬 처리 엔진 ================
def process_csv_file(client, input_path, output_path, max_workers=8, requests_per_minute=60):
    """
    병렬 처리 + 중간 저장 + 동적 지연
    max_workers: 동시 요청 수 (유료 티어에서는 8~10 권장)
    requests_per_minute: 분당 요청 한도 (유료 티어는 60 이상)
    """
    try:
        df = pd.read_csv(input_path, engine='python')
        print(f"📂 파일 로드 완료: {len(df)}개 행")

        if 'content' not in df.columns:
            raise KeyError("'content' 컬럼이 존재하지 않습니다.")

        total_rows = len(df)
        batch_size = max_workers
        interval = 60.0 / requests_per_minute * batch_size  # ex) 8개/60회 = 8초

        processed_count = 0
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                futures = []
                for idx in range(batch_start, batch_end):
                    original = df.at[idx, 'content']
                    future = executor.submit(generate_summary, client, original)
                    futures.append((idx, future))

                for idx, future in futures:
                    try:
                        result = future.result()
                        df.at[idx, 'content'] = result
                        processed_count += 1

                        # 진행률 표시
                        if processed_count % 10 == 0:
                            print(f"진행률: {processed_count}/{total_rows} ({processed_count/total_rows*100:.1f}%)")

                        # 100행마다 중간 저장
                        if processed_count % 100 == 0:
                            df.to_csv(output_path, index=False, encoding='utf-8-sig')
                            print(f"💾 {processed_count}행 처리 후 임시 저장 완료")
                    except Exception as e:
                        print(f"⚠️ 행 {idx} 처리 실패: {str(e)}")
                        df.at[idx, 'content'] = f"오류: {str(e)}"

                # 배치 간 동적 대기
                elapsed = time.time() - start_time
                sleep_time = max(interval - elapsed, 0)
                if batch_end < total_rows:
                    time.sleep(sleep_time)
                start_time = time.time()

        # 최종 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 최종 저장 완료: {output_path}")

        # 결과 샘플 출력
        print("\n=== 처리 결과 미리보기 ===")
        for i in range(min(3, len(df))):
            print(f"[요약 {i+1}] {df.iloc[i]['content'][:100]}...")

    except Exception as e:
        print(f"❌ CSV 처리 오류: {str(e)}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"⚠️ 오류 발생! 현재까지 처리한 내용 저장 완료: {output_path}")

# ================ 메인 실행 ================
if __name__ == "__main__":
    INPUT_PATH = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\bill_filtered_final.csv"
    OUTPUT_PATH = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\processed_bills_optimized.csv"

    gemini_client = initialize_system()
    if gemini_client:
        process_csv_file(
            gemini_client,
            INPUT_PATH,
            OUTPUT_PATH,
            max_workers=8,             # 유료 티어 기준
            requests_per_minute=60
        )
