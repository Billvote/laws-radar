import os
import re
import pandas as pd
import olefile
import zlib
import struct
import csv
from multiprocessing import Pool, cpu_count

# 경로 설정
hwp_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/HWP/22'
output_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/HWP_summary'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'law_final22_1.csv')

if not os.path.exists(hwp_dir):
    raise FileNotFoundError(f"❌ HWP 폴더가 존재하지 않습니다: {hwp_dir}")

hwp_files = [f for f in os.listdir(hwp_dir) if f.lower().endswith('.hwp')]
if not hwp_files:
    raise FileNotFoundError(f"❌ HWP 폴더에 HWP 파일이 없습니다: {hwp_dir}")

def preprocess_text(text):
    text = re.sub(r'[※■◆▷▶●△○◎☆✓ⓒ═━─—_]+', '', text)
    text = re.sub(r'[\n\r]+', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def oneline(text):
    text = re.sub(r'(제안연월일|발의연월일|제출연월일|제출일|제안일)[ :：\-]*[\d\.]*,?', '', text)
    text = re.sub(r'[═━─—_]+', '', text)
    text = text.replace('"', '')
    text = text.replace(',', ' ')
    text = re.sub(r'[\n\r\t]+', ' ', text)
    return text.strip()

def extract_text_from_hwp(hwp_path):
    """
    HWP 파일에서 전체 텍스트를 추출하는 함수
    olefile과 zlib를 사용하여 Body Sections에서 직접 텍스트 추출
    """
    try:
        f = olefile.OleFileIO(hwp_path)
        dirs = f.listdir()
        
        # HWP 파일 검증
        if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
            print(f"유효하지 않은 HWP 파일: {hwp_path}")
            return ""
        
        # 문서 포맷 압축 여부 확인
        header = f.openstream("FileHeader")
        header_data = header.read()
        is_compressed = (header_data[0] & 1) == 1
        
        # Body Sections 불러오기
        nums = []
        for d in dirs:
            if len(d) > 1 and d[0] == "BodyText" and d[1].startswith("Section"):
                nums.append(int(d[1][len("Section"):]))
        
        sections = ["BodyText/Section" + str(x) for x in sorted(nums)]
        
        # 전체 text 추출
        text = ""
        for section in sections:
            try:
                bodytext = f.openstream(section)
                data = bodytext.read()
                
                if is_compressed:
                    unpacked_data = zlib.decompress(data, -15)
                else:
                    unpacked_data = data
                
                # 각 Section 내 text 추출
                section_text = ""
                i = 0
                size = len(unpacked_data)
                
                while i < size:
                    if i + 4 > size:
                        break
                    
                    header = struct.unpack_from("<I", unpacked_data, i)[0]
                    rec_type = header & 0x3ff
                    rec_len = (header >> 20) & 0xfff
                    
                    if rec_type in [67]:  # 텍스트 레코드 타입
                        if i + 4 + rec_len <= size:
                            rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                            try:
                                section_text += rec_data.decode('utf-16le')
                                section_text += "\n"
                            except UnicodeDecodeError:
                                continue
                    
                    i += 4 + rec_len
                
                text += section_text
                text += "\n"
                
            except Exception as e:
                print(f"섹션 처리 오류 {section}: {e}")
                continue
        
        f.close()
        return preprocess_text(text)
        
    except Exception as e:
        print(f"HWP 읽기 오류: {hwp_path} ({str(e)})")
        return ""

# 정규표현식 패턴들 (기존과 동일)
pattern_제안이유 = r'(?:^|\n)\s*\d*\.?\s*제안[ ]?이유[ \t]*[\n:：\-]*([\s\S]*?)(?=(\n[^\n]{0,10}(주요[ ]?내용|주문)[ \t]*[\n:：\-]|$|\n[^\n]{0,20}(참고사항|제안자|제출연월일|부칙)))'
pattern_주요내용 = r'(?:^|\n)\s*\d*\.?\s*주요[ ]?내용[ \t]*[\n:：\-]*([\s\S]*?)(?=(\n[^\n]{0,20}(참고사항|주문|파견연장 동의안|비용추계서|제안이유|$)))'
pattern_주문 = r'(?:^|\n)\s*\d*\.?\s*주[ ]?문[ \t]*[\n:：\-]*([\s\S]*?)(?=(\n[^\n]{0,20}(참고사항|부칙|제안자|시행일|$)))'

def extract_legal_sections(text):
    sections = {
        '의안번호': '',
        '제안이유': '',
        '주요내용': '',
        '주문': ''
    }
    
    m = re.search(r'(의안\s*번호|의안번호)[ :\-]*([^\n\\]+)', text)
    if m:
        sections['의안번호'] = m.group(2).strip()
    
    m = re.search(pattern_제안이유, text)
    if m:
        sections['제안이유'] = m.group(1).strip()
    
    m = re.search(pattern_주요내용, text)
    if m:
        sections['주요내용'] = m.group(1).strip()
    
    m = re.search(pattern_주문, text)
    if m:
        sections['주문'] = m.group(1).strip()
    
    return sections

def process_file(filename):
    try:
        hwp_path = os.path.join(hwp_dir, filename)
        text = extract_text_from_hwp(hwp_path)
        
        if not text.strip():
            print(f"텍스트 추출 실패: {filename}")
            return None

        sections = extract_legal_sections(text)
        
        if not sections['제안이유'] and not sections['주요내용'] and not sections['주문']:
            print(f"제안이유/주요내용/주문 모두 없음: {filename} → PASS")
            return None

        return {
            '의안번호': oneline(sections['의안번호']) if sections['의안번호'] else pd.NA,
            '제안이유': oneline(sections['제안이유']) if sections['제안이유'] else pd.NA,
            '주요내용': oneline(sections['주요내용']) if sections['주요내용'] else pd.NA,
            '주문': oneline(sections['주문']) if sections['주문'] else pd.NA
        }
    except Exception as e:
        print(f"오류 발생: {filename} ({str(e)})")
        return None

def process_hwps_to_csv():
    print(f"🔍 {len(hwp_files)}개 HWP 병렬 처리 시작...")
    
    # CPU 코어 수의 2배로 워커 설정 (최대 8개로 제한)
    num_workers = min(8, max(4, cpu_count() * 2))
    
    with Pool(num_workers) as pool:
        results = list(filter(None, pool.map(process_file, hwp_files)))

    if not results:
        print("❌ 처리된 HWP가 없습니다. 폴더와 파일을 다시 확인하세요.")
        return

    df = pd.DataFrame(results, columns=['의안번호', '제안이유', '주요내용', '주문'])
    df.to_csv(output_csv, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"\n✅ 최종 CSV 저장: {output_csv}")
    print(f"총 처리 파일: {len(results)}개")

if __name__ == '__main__':
    # 필요한 라이브러리 설치 확인
    try:
        import olefile
    except ImportError:
        print("❌ olefile 라이브러리가 필요합니다. 다음 명령으로 설치하세요:")
        print("pip install olefile")
        exit(1)
    
    process_hwps_to_csv()
