import pandas as pd
import os
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

def initialize_system():
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

def insert_possessive_eui(phrase):
    if '의' in phrase:
        return phrase
    tokens = phrase.split()
    possessive_targets = ['환자', '공원', '학교', '기관', '센터', '학생', '교사', '어린이', '노인', '아동', '장애인']
    if len(tokens) >= 2 and tokens[-2] in possessive_targets:
        tokens.insert(-1, '의')
        return ' '.join(tokens)
    return phrase

def has_jongseong(word):
    if not word:
        return False
    code = ord(word[-1])
    return (code - 44032) % 28 != 0

def get_last_word(phrase):
    tokens = [t for t in phrase.strip().split() if re.match(r'^[가-힣]+$', t)]
    return tokens[-1] if tokens else phrase

def correct_spacing_and_spell(text):
    try:
        from pykospacing import Spacing
        from hanspell import spell_checker
        spacing = Spacing()
        spaced_text = spacing(text)
        spelled_text = spell_checker.check(spaced_text).checked
        spelled_text = re.sub(r'([가-힣]+)(은|는|이|가|을|를|의|에|에서|으로|로){2,}', r'\1\2', spelled_text)
        spelled_text = re.sub(r'([가-힣]+)(은|는) ', lambda m: m.group(1) + ('은 ' if has_jongseong(m.group(1)) else '는 '), spelled_text)
        spelled_text = re.sub(r'(\w+) 에 서', r'\1에서', spelled_text)
        spelled_text = re.sub(r'(\w+) 으로', r'\1로', spelled_text)
        spelled_text = re.sub(r'(\w+) 을', r'\1를', spelled_text)
        return spelled_text
    except ImportError:
        text = re.sub(r'\s+의\s+', '의 ', text)
        text = re.sub(r'\s+([은는이가을를에에서으로로])\s+', r'\1 ', text)
        return text

def replace_comma_with_dot(text):
    if isinstance(text, str):
        return text.replace(",", "·")
    return text

def josa(word, josa_pair):
    if not word:
        return josa_pair[1]
    return josa_pair[0] if has_jongseong(word) else josa_pair[1]

def postprocess_korean_sentence(p1, p2, p3):
    p1 = insert_possessive_eui(p1)
    last_word = get_last_word(p1)

    method_nouns = [
        "신설", "강화", "확대", "도입", "분리", "지원", "완화", "정비", "확립", "정착", "개정", "통합", "폐지", "보완", "운영", "설치", "변경"
    ]
    purpose_nouns = ["개선", "예방", "방지", "달성", "실현", "확보", "정착", "유도", "촉진", "해소"]

    # p3가 method_nouns로 끝나면 목적어+을/를+동사+함으로써 개선하고자 함
    for noun in method_nouns:
        if p3.strip().endswith(noun):
            p3_tokens = p3.strip().split()
            if len(p3_tokens) > 1:
                obj = ' '.join(p3_tokens[:-1])
                verb = p3_tokens[-1]
                obj_josa = josa(get_last_word(obj), ("을", "를"))
                return f"{p1}{josa(last_word, ('은', '는'))} {p2}에서 비롯되어 {obj}{obj_josa} {verb}함으로써 개선하고자 함"
            else:
                # 단일어만 있을 때
                return f"{p1}{josa(last_word, ('은', '는'))} {p2}에서 비롯되어 {p3.strip()}함으로써 개선하고자 함"
    for noun in purpose_nouns:
        if p3.strip().endswith(noun):
            조사3 = josa(p3.strip(), ("으로", "로"))
            return f"{p1}{josa(last_word, ('은', '는'))} {p2}에서 비롯되어 {p3}{조사3} 하고자 함"
    조사3 = josa(p3.strip(), ("으로", "로"))
    return f"{p1}{josa(last_word, ('은', '는'))} {p2}에서 비롯되어 {p3}{조사3} 개선하고자 함"

def convert_to_sentence(summary):
    parts = [p.split(": ")[1].strip() if ": " in p else p.strip()
             for p in summary.split("→")]
    if len(parts) != 3:
        return summary.replace(",", "·")
    p1, p2, p3 = parts
    return postprocess_korean_sentence(p1, p2, p3)

def generate_summary(client, original_text, max_retries=5):
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
            final_text = convert_to_sentence(summary)
            final_text = correct_spacing_and_spell(final_text)
            final_text = replace_comma_with_dot(final_text)
            return final_text
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

def process_csv_file(client, input_path, output_path, max_workers=8, requests_per_minute=60):
    try:
        df = pd.read_csv(input_path, engine='python')
        print(f"📂 파일 로드 완료: {len(df)}개 행")

        if 'content' not in df.columns:
            raise KeyError("'content' 컬럼이 존재하지 않습니다.")

        total_rows = len(df)
        batch_size = max_workers
        interval = 60.0 / requests_per_minute * batch_size

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
                        result = replace_comma_with_dot(result)
                        df.at[idx, 'content'] = result
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"진행률: {processed_count}/{total_rows} ({processed_count/total_rows*100:.1f}%)")
                        if processed_count % 100 == 0:
                            df['content'] = df['content'].apply(replace_comma_with_dot)
                            df.to_csv(output_path, index=False, encoding='utf-8-sig')
                            print(f"💾 {processed_count}행 처리 후 임시 저장 완료")
                    except Exception as e:
                        print(f"⚠️ 행 {idx} 처리 실패: {str(e)}")
                        df.at[idx, 'content'] = f"오류: {str(e)}"

                elapsed = time.time() - start_time
                sleep_time = max(interval - elapsed, 0)
                if batch_end < total_rows:
                    time.sleep(sleep_time)
                start_time = time.time()

        df['content'] = df['content'].apply(replace_comma_with_dot)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 최종 저장 완료: {output_path}")

        print("\n=== 처리 결과 미리보기 ===")
        for i in range(min(3, len(df))):
            print(f"[요약 {i+1}] {df.iloc[i]['content'][:100]}...")

    except Exception as e:
        print(f"❌ CSV 처리 오류: {str(e)}")
        df['content'] = df['content'].apply(replace_comma_with_dot)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"⚠️ 오류 발생! 현재까지 처리한 내용 저장 완료: {output_path}")

if __name__ == "__main__":
    INPUT_PATH = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\bill_filtered_final.csv"
    OUTPUT_PATH = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\processed_bills_optimized_final7.csv"

    gemini_client = initialize_system()
    if gemini_client:
        process_csv_file(
            gemini_client,
            INPUT_PATH,
            OUTPUT_PATH,
            max_workers=8,
            requests_per_minute=60
        )
