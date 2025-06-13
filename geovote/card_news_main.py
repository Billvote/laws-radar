import pandas as pd
import google.generativeai as genai
import time
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CardNewsConverter:
    def __init__(self, api_key: str):
        self.api_key = api_key
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
        """최적화된 프롬프트 - 하나의 문구만 생성"""
        return f"""
다음 법률/정책 내용을 카드뉴스용 문구로 변환해주세요.

요구사항:
- 20자 이내
- 시민 친화적 표현
- 혜택이나 개선사항 강조
- 하나의 문구만 출력

변환할 내용: {content}

최적화된 문구:"""

    def convert_single(self, content: str, retry_count: int = 3) -> str:
        """단일 내용 변환"""
        if not content or pd.isna(content):
            return "내용 없음"
        
        prompt = self._create_optimized_prompt(str(content))
        
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.5,
                        max_output_tokens=50,
                        top_p=0.8,
                    )
                )
                
                if response and response.text:
                    result = response.text.strip()
                    # 불필요한 텍스트 제거
                    result = result.replace("**", "").replace("옵션", "").replace("선택", "")
                    result = result.split('\n')[0]  # 첫 번째 줄만 사용
                    result = result.split('**')[0]  # ** 이전 텍스트만 사용
                    
                    if len(result) > 5 and len(result) <= 25:
                        logger.info(f"✅ 변환 성공: {result}")
                        return result
                
                logger.warning(f"⚠️ 부적절한 응답 ({attempt + 1}/{retry_count})")
                
            except Exception as e:
                logger.warning(f"⚠️ API 호출 실패 ({attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)
        
        # 실패시 수동 변환
        return self._manual_convert(content)
    
    def _manual_convert(self, content: str) -> str:
        """API 실패시 수동 변환"""
        content = str(content)
        
        if "입국 불허" in content:
            return "안전한 대한민국, 확실하게 지켜요!"
        elif "조합 임원" in content and "벌금형" in content:
            return "조합 임원 처벌 강화로 공정성 UP"
        elif "공직선거법" in content and "방송광고" in content:
            return "깨끗한 선거, 투명한 방송광고!"
        elif "법적 근거" in content:
            return "법적 기반 마련으로 제도 개선"
        elif "강화" in content:
            return "보안 강화로 더 안전하게!"
        else:
            return "제도 개선으로 더 나은 사회!"

    def test_api_connection(self):
        """API 연결 테스트"""
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
        """CSV 파일 처리"""
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
    API_KEY = "AIzaSyA8M00iSzCK1Lvc5YfxamYgQf-Lh4xh5R0"
    
    INPUT_FILE = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\summary_of_content_short.csv"
    OUTPUT_FILE = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\card_news_output.csv"
    
    try:
        print("🎯 카드뉴스 변환기 시작")
        
        converter = CardNewsConverter(API_KEY)
        result_df = converter.process_csv(INPUT_FILE, OUTPUT_FILE)
        
        print("\n✅ 변환 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
