import os
import re
import pandas as pd
from PyPDF2 import PdfReader
import csv
from multiprocessing import Pool, cpu_count

# 경로 설정
pdf_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF/22'
output_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF_summary'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'law_final22_1.csv')

if not os.path.exists(pdf_dir):
    raise FileNotFoundError(f"❌ PDF 폴더가 존재하지 않습니다: {pdf_dir}")

pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
if not pdf_files:
    raise FileNotFoundError(f"❌ PDF 폴더에 PDF 파일이 없습니다: {pdf_dir}")

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

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return '\n'.join([preprocess_text(page.extract_text() or '') for page in reader.pages])
    except Exception as e:
        print(f"PDF 읽기 오류: {pdf_path} ({str(e)})")
        return ""

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
        pdf_path = os.path.join(pdf_dir, filename)
        text = extract_text_from_pdf(pdf_path)
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

def process_pdfs_to_csv():
    print(f"🔍 {len(pdf_files)}개 PDF 병렬 처리 시작...")
    
    # CPU 코어 수의 2배로 워커 설정 (최대 8개로 제한)
    num_workers = min(8, max(4, cpu_count() * 2))
    
    with Pool(num_workers) as pool:
        results = list(filter(None, pool.map(process_file, pdf_files)))

    if not results:
        print("❌ 처리된 PDF가 없습니다. 폴더와 파일을 다시 확인하세요.")
        return

    df = pd.DataFrame(results, columns=['의안번호', '제안이유', '주요내용', '주문'])
    df.to_csv(output_csv, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"\n✅ 최종 CSV 저장: {output_csv}")
    print(f"총 처리 파일: {len(results)}개")

if __name__ == '__main__':
    process_pdfs_to_csv()
