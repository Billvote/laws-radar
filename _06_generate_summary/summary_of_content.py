# 의안내용 요약하는 코드

import sys
from pathlib import Path

# 현재 파일의 상위(부모) 디렉토리(laws-radar)를 sys.path에 추가 (상대경로 방식)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import time
import random
import re
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from google.generativeai import types
from settings import GEOVOTE_DATA_DIR  # settings.py에서 GEOVOTE_DATA_DIR 임포트

# --- 환경변수 및 dotenv 적용 부분 추가 ---
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 불러오기

class GracefulExiter:
    def __init__(self):
        self.exit = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\n⚠️ 안전한 종료를 시작합니다. 현재까지 처리된 내용을 저장합니다...")
        self.exit = True

def initialize_system():
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API 초기화 완료")
        return genai
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {str(e)}")
        return None

def emergency_save(df, path):
    try:
        df.to_csv(str(path) + ".emergency", index=False, encoding='utf-8-sig')
        print(f"🆘 비상 저장 완료: {path}.emergency")
    except Exception as e:
        print(f"❌ 비상 저장 실패: {str(e)}")

def correct_spacing_and_spell(text):
    try:
        from pykospacing import Spacing
        from hanspell import spell_checker
        spacing = Spacing()
        spaced_text = spacing(text)
        spelled_text = spell_checker.check(spaced_text).checked
        return spelled_text
    except ImportError:
        return text
    except Exception:
        return text

def convert_to_sentence(summary):
    text = re.sub(r'[→·|,]', '', summary).strip()
    if not re.search(r'(함|강화함|개선함|마련함|중단함|추진함|신설함|삭제함|요)$', text):
        text += '함'
    return text

def parse_gemini_response(response):
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts and hasattr(parts[0], 'text'):
                    return parts[0].text.strip()
        if hasattr(response, 'text'):
            return response.text.strip()
        if hasattr(response, 'result'):
            result = response.result
            if hasattr(result, 'candidates') and result.candidates:
                candidate = result.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and hasattr(parts[0], 'text'):
                        return parts[0].text.strip()
        if hasattr(response, '_result'):
            result = response._result
            if hasattr(result, 'candidates') and result.candidates:
                candidate = result.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and hasattr(parts[0], 'text'):
                        return parts[0].text.strip()
        response_str = str(response)
        if response_str and response_str != str(type(response)):
            return response_str.strip()
        print("🔴 응답 구조 분석:")
        print(f"응답 타입: {type(response)}")
        print(f"응답 속성: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        return None
    except AttributeError as e:
        print(f"🔴 속성 오류: {str(e)}")
        return None
    except Exception as e:
        print(f"🔴 파싱 오류: {str(e)}")
        return None

def generate_summary(client, original_text, max_retries=5):
    if pd.isna(original_text) or not original_text.strip():
        return original_text

    for attempt in range(max_retries):
        try:
            model = client.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"""
[법안 요약 규칙]
1. 반드시 한 문장으로, 핵심 조치와 그 원인(이유, 배경 등)이 모두 포함되도록 간결하게 요약하세요.
2. 요약문은 항상 '~을/를 [원인/배경/이유]로 인해/에서 기인하여 ~을/를 ...함' 또는 '~을/를 ...함'의 형태로 끝나도록 통일하세요.
3. '개선함', '강화함', '중단함', '마련함', '추진함', '신설함', '삭제함' 등 정책적 어미를 사용하세요.
4. 불필요한 수식어, 반복, 배경설명, '관련 개선 방안' 등 넣지 마세요.
5. 특수문자(→, ·, |, ,) 사용 금지.
6. 예시:
- 인사 관련 제도를 복잡한 절차로 인해 개선함
- 지원 제외 규정을 남용 사례 증가로 강화함
- 불필요한 위원회 운영을 실효성 부족으로 중단함
- 공사명 변경의 혼란을 방지하기 위해 관련 법안을 마련함

[원문]
{original_text}

[요약]
"""
            response = model.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=100
                )
            )
            summary = parse_gemini_response(response)
            if not summary:
                raise ValueError("Gemini 응답에서 요약 텍스트를 찾을 수 없음")
            final_text = convert_to_sentence(summary)
            final_text = correct_spacing_and_spell(final_text)
            final_text = re.sub(r'[→·|,]', '', final_text)
            return final_text
        except Exception as e:
            err_msg = str(e)
            print(f"🔴 시도 {attempt + 1}/{max_retries} 실패: {err_msg}")
            if any(code in err_msg for code in ["503", "UNAVAILABLE", "SERVICE_UNAVAILABLE"]):
                wait = random.randint(10, 30) * (attempt + 1)
                print(f"⚠️ 서버 오류: {wait}초 후 재시도")
                time.sleep(wait)
            elif "429" in err_msg or "RATE_LIMIT" in err_msg:
                print(f"⏳ 요청 한도 초과: 60초 대기")
                time.sleep(60)
            else:
                wait = random.randint(3, 8)
                print(f"⚠️ 일반 오류: {wait}초 후 재시도")
                time.sleep(wait)
    print(f"❌ 최대 재시도 횟수 초과, 원본 텍스트 유지")
    return original_text
# 추가
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_summaries_parallel(df, client, max_workers=5):
    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_summary, client, df.at[i, "content"]): i
            for i in range(len(df))
        }

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="요약 생성 중"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"인덱스 {idx} 처리 실패: {e}")
                results[idx] = df.at[idx, "content"]  # 실패 시 원본 유지

    df["summary"] = results
    return df

# ---
def process_csv_file(client, input_path, output_path, max_workers=4, requests_per_minute=30):
    exiter = GracefulExiter()
    df = pd.read_csv(input_path, engine='python')
    print(f"📂 파일 로드 완료: {len(df)}개 행")
    atexit.register(emergency_save, df.copy(), output_path)
    try:
        total_rows = len(df)
        batch_size = max_workers
        interval = 60.0 / requests_per_minute * batch_size
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, total_rows, batch_size):
                if exiter.exit:
                    print("🚨 사용자 요청에 의해 처리 중단")
                    executor.shutdown(wait=False)
                    break
                batch_end = min(batch_start + batch_size, total_rows)
                futures = [(idx, executor.submit(generate_summary, client, df.at[idx, 'content']))
                          for idx in range(batch_start, batch_end)]
                for idx, future in futures:
                    if exiter.exit:
                        break
                    try:
                        result = future.result(timeout=120)
                        df.at[idx, 'content'] = result
                    except Exception as e:
                        print(f"⚠️ 행 {idx} 처리 실패: {str(e)}")
                        df.at[idx, 'content'] = f"오류: {str(e)}"
                processed = min(batch_end, total_rows)
                print(f"진행률: {processed}/{total_rows} ({processed/total_rows*100:.1f}%)")
                if batch_end % 20 == 0:
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    print(f"💾 임시 저장 완료: {batch_end}행")
                elapsed = time.time() - start_time
                sleep_time = max(interval - elapsed, 0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                start_time = time.time()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 최종 저장 완료: {output_path}")
    except Exception as e:
        print(f"❌ CSV 처리 오류: {str(e)}")
    finally:
        atexit.unregister(emergency_save)
        emergency_save(df, output_path)

if __name__ == "__main__":
    INPUT_PATH = GEOVOTE_DATA_DIR / "bill_filtered_final.csv"
    OUTPUT_PATH = GEOVOTE_DATA_DIR / "summary_of_content.csv"
    gemini_client = initialize_system()
    if gemini_client:
        process_csv_file(
            gemini_client,
            INPUT_PATH,
            OUTPUT_PATH,
            max_workers=4,
            requests_per_minute=30
        )
