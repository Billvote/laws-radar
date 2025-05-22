# 원안 pdf의 제안이유, 주요내용만 추출 => PDF_summary 폴더에 저장
# 제안이유, 주요내용 없는건 추출안함

import os
import re
import pandas as pd
from PyPDF2 import PdfReader
import csv

# PDF 파일이 들어있는 폴더
pdf_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF/20'

# 결과 CSV를 저장할 폴더 및 경로
output_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF_summary'
os.makedirs(output_dir, exist_ok=True)  # 결과 폴더 없으면 생성
output_csv = os.path.join(output_dir, 'law_final20.csv')

# PDF 폴더 및 파일 체크
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
    # 날짜 라벨 및 값 제거
    text = re.sub(r'(제안연월일|발의연월일|제출연월일|제출일|제안일)[ :：\-]*[\d\.]*,?', '', text)
    text = re.sub(r'[═━─—_]+', '', text)
    text = text.replace('"', '')  # 쌍따옴표 제거
    text = text.replace(',', ' ') # 쉼표도 공백으로 변환 (권장)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return '\n'.join([preprocess_text(page.extract_text() or '') for page in reader.pages])
    except Exception as e:
        print(f"PDF 읽기 오류: {pdf_path} ({str(e)})")
        return ""

# 번호(2. 등)와 띄어쓰기(제안 이유 등) 모두 허용하는 패턴
pattern_제안이유 = r'(?:^|\n)\s*\d*\.?\s*제안[ ]?이유[ \t]*[\n:：\-]*([\s\S]*?)(?=(\n[^\n]{0,10}주요[ ]?내용[ \t]*[\n:：\-]|$|\n[^\n]{0,20}(참고사항|제안자|제출연월일|부칙)))'
pattern_주요내용 = r'(?:^|\n)\s*\d*\.?\s*주요[ ]?내용[ \t]*[\n:：\-]*([\s\S]*?)(?=(\n[^\n]{0,20}(참고사항|파견연장 동의안|비용추계서|제안이유|$)))'

def extract_legal_sections(text):
    sections = {
        '의안번호': '',
        '제안이유': '',
        '주요내용': ''
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
    return sections

def process_pdfs_to_csv():
    all_data = []
    processed_count = 0
    skipped_count = 0

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"텍스트 추출 실패: {filename}")
            skipped_count += 1
            continue

        sections = extract_legal_sections(text)
        # 둘 다 없으면 패스 (CSV에 포함X)
        if not sections['제안이유'] and not sections['주요내용']:
            print(f"제안이유/주요내용 모두 없음: {filename} → PASS")
            skipped_count += 1
            continue

        row = {
            '의안번호': oneline(sections['의안번호']) if sections['의안번호'] else pd.NA,
            '제안이유': oneline(sections['제안이유']) if sections['제안이유'] else pd.NA,
            '주요내용': oneline(sections['주요내용']) if sections['주요내용'] else pd.NA
        }
        all_data.append(row)
        processed_count += 1

    if not all_data:
        print("❌ 처리된 PDF가 없습니다. 폴더와 파일을 다시 확인하세요.")
        return

    df = pd.DataFrame(all_data, columns=['의안번호', '제안이유', '주요내용'])
    # quoting=csv.QUOTE_NONE: 쌍따옴표 없이 저장
    df.to_csv(output_csv, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"\n✅ 최종 CSV 저장: {output_csv}")
    print(f"총 처리 파일: {processed_count}개, 건너뛴 파일: {skipped_count}개")

if __name__ == '__main__':
    process_pdfs_to_csv()
