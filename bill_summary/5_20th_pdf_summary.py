import os
import re
import fitz  # PyMuPDF
import pandas as pd
from multiprocessing import Pool, cpu_count

# 경로 설정
pdf_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF/20'
output_dir = r'C:/Users/1-02/Desktop/DAMF2/laws-radar/bill_summary/PDF_summary'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'law_final20.csv')

# Tesseract OCR 경로 설정 (필요시 변경)
tessdata_dir = r'C:/Program Files/Tesseract-OCR/tessdata'
os.environ['TESSDATA_PREFIX'] = tessdata_dir

pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

def ocr_text_extraction(page):
    """OCR을 이용한 텍스트 추출"""
    pix = page.get_pixmap(dpi=300)
    img_path = "temp_ocr.png"
    pix.save(img_path)
    text = fitz.utils.ocr_page(page, language="kor+eng")  # 한글+영어 OCR
    os.remove(img_path)
    return text

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page in doc:
        # 먼저 일반 텍스트 추출 시도
        text = page.get_text()
        if not text.strip():  # 텍스트가 없으면 OCR 수행
            text = ocr_text_extraction(page)
        full_text.append(text)
    return "\n".join(full_text)

def extract_info(text):
    # 의안번호 추출 (패턴 강화)
    bill_no = re.search(r'의안\s*번호[\s:：-]*([가-힣\d-]+)', text)
    bill_no = bill_no.group(1).strip() if bill_no else ''
    
    # 제안이유 추출 (종료 패턴 확장)
    reason = re.search(r'제안\s*이유[\s:：-]*([\s\S]*?)(?=\n\s*(주요\s*내용|참고\s*사항|부칙|제안자|결론|$))', text)
    reason = reason.group(1).strip() if reason else ''
    
    # 주요내용 추출 (종료 패턴 확장)
    content = re.search(r'주요\s*내용[\s:：-]*([\s\S]*?)(?=\n\s*(참고\s*사항|부칙|제안자|결론|$))', text)
    content = content.group(1).strip() if content else ''
    
    return bill_no, reason, content

def process_file(filename):
    try:
        pdf_path = os.path.join(pdf_dir, filename)
        text = process_pdf(pdf_path)
        bill_no, reason, content = extract_info(text)
        return {
            '파일명': filename,
            '의안번호': re.sub(r'\s+', ' ', bill_no),
            '제안이유': re.sub(r'\s+', ' ', reason),
            '주요내용': re.sub(r'\s+', ' ', content)
        }
    except Exception as e:
        return {'파일명': filename, '의안번호': '', '제안이유': '', '주요내용': '', '오류': str(e)}

def main():
    print(f"🔍 {len(pdf_files)}개 PDF 처리 시작...")
    with Pool(max(4, cpu_count())) as pool:
        results = pool.map(process_file, pdf_files)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV 저장 완료: {output_csv}")

    # 누락 정보 추적
    no_bill_files = []
    no_content_files = []
    error_files = []

    for res in results:
        if '오류' in res and res['오류']:
            error_files.append(res['파일명'])
        else:
            # 의안번호 없는 파일
            if not res['의안번호'].strip():
                no_bill_files.append(res['파일명'])
            # 제안이유/주요내용 모두 없는 파일
            if not res['제안이유'].strip() and not res['주요내용'].strip():
                no_content_files.append(res['파일명'])

    # 리포트 출력
    print("\n📊 처리 결과 리포트")
    print(f"✅ 성공 처리: {len(results) - len(error_files)}개")
    print(f"⏩ 제안이유/주요내용 모두 없는 파일: {len(no_content_files)}개")
    print(f"❌ 오류: {len(error_files)}개")

    if no_bill_files:
        print(f"\n⚠️ 의안번호 없는 파일 ({len(no_bill_files)}개):")
        for f in no_bill_files:
            print(f"   - {f}")

    if no_content_files:
        print(f"\n⚠️ 제안이유/주요내용 모두 없는 파일 ({len(no_content_files)}개):")
        for f in no_content_files:
            print(f"   - {f}")

    if error_files:
        print(f"\n⚠️ 오류 발생 파일 ({len(error_files)}개):")
        for f in error_files:
            print(f"   - {f}")

if __name__ == '__main__':
    main()
