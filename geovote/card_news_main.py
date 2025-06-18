# -*- coding: utf-8 -*-
import os
import pandas as pd
import google.generativeai as genai
import time
import logging
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CardNewsConverter:
    def __init__(self, api_key: str = None):
        # API 키를 인자로 받거나 환경 변수에서 로드
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API 키가 제공되지 않았습니다. 인자로 전달하거나 .env 파일에 GEMINI_API_KEY를 설정하세요.")
        
        self.model = None
        self._setup_api()
    
    def _setup_api(self):
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("✅ Gemini API 설정 완료")
        except Exception as e:
            logger.error(f"❌ API 설정 실패: {e}")
            raise

    def _create_optimized_prompt(self, content: str) -> str:
        # 예시 답변을 포함한 프롬프트로 개선
        return f"""
아래의 법률/정책 내용을 카드뉴스용 한 줄 메시지(20자 내외, 시민 친화적, 혜택·개선 강조, 느낌표 등 감탄사 포함)로 바꿔주세요.

예시:
1. 조합 임원, 벌금형 분리 선고로 공정성 UP!
2. 우리 아이들의 안전을 위해! 어린이공원 교통안전 강화!
3. 형사처벌 공정성 확보! 벌금형 기준 현실화!
4. 한국마사회, 농업인 위한 공간으로 변신!
5. 경우회 목적 및 운영 투명성 강화!

변환할 내용: {content}

카드뉴스 문구:
"""

    def convert_single(self, content: str, retry_count: int = 3) -> str:
        if not content or pd.isna(content):
            return "내용 없음"
        
        prompt = self._create_optimized_prompt(str(content))
        
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.9,  # 더 창의적으로
                        max_output_tokens=100,  # 충분히 길게
                        top_p=1.0,
                    )
                )
                
                if response and response.text:
                    result = response.text.strip()
                    # 불필요한 **, 번호, "카드뉴스 문구:" 등만 제거
                    result = result.replace("**", "").replace("카드뉴스 문구:", "").strip()
                    # 2줄 이상이면 첫 줄만 사용
                    result = result.split('\n')[0].strip()
                    # 너무 짧거나 길면 재시도
                    if 7 <= len(result) <= 35:
                        logger.info(f"✅ 변환 성공: {result}")
                        return result
                
                logger.warning(f"⚠️ 부적절한 응답 ({attempt + 1}/{retry_count})")
                
            except Exception as e:
                logger.warning(f"⚠️ API 호출 실패 ({attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)
        
        return self._manual_convert(content)
    
    def _manual_convert(self, content: str) -> str:
        # 필요시 직접 작성
        return "더 나은 사회를 위한 변화!"

    def test_api_connection(self):
        try:
            test_response = self.model.generate_content("테스트")
            if test_response and test_response.text:
                logger.info("✅ API 연결 테스트 성공")
                return True
            else:
                logger.error("❌ API 응답이 비어있음")
                return False
        except Exception as e:
            logger.error(f"❌ API 연결 테스트 실패: {e}")
            return False

    def process_csv(self, input_path: str, output_path: str, delay: float = 1.5):
        try:
            if not self.test_api_connection():
                raise Exception("API 연결에 문제가 있습니다.")
            
            logger.info(f"📂 CSV 파일 읽는 중: {input_path}")
            df = pd.read_csv(input_path, encoding='utf-8')
            
            if 'content' not in df.columns:
                raise ValueError(f"'content' 컬럼이 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
            
            contents = df['content'].tolist()
            results = []
            
            logger.info(f"🚀 총 {len(contents)}개 항목 변환 시작")
            
            for idx, content in enumerate(contents, 1):
                logger.info(f"📝 처리 중: {idx}/{len(contents)}")
                
                result = self.convert_single(content)
                results.append(result)
                
                logger.info(f"결과: {result}")
                
                if idx < len(contents):
                    time.sleep(delay)
            
            df['card_news_content'] = results
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"✅ 변환 완료! 결과 저장: {output_path}")
            return df
            
        except Exception as e:
            logger.error(f"❌ 처리 중 오류: {e}")
            raise

# ===== 메인 실행 =====
if __name__ == "__main__":
    # 환경 변수에서 API 키 로드 (기본값은 None)
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "summary_of_content_short.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "card_news_output.csv")
    
    try:
        print("🎯 카드뉴스 변환기 시작")
        
        # API 키가 환경 변수에 없으면 오류 발생
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        converter = CardNewsConverter(API_KEY)
        result_df = converter.process_csv(INPUT_FILE, OUTPUT_FILE)
        
        print("\n✅ 변환 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
