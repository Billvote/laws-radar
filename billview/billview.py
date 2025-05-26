import os
import pandas as pd
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import re

# 프로젝트 루트 경로 (billview의 상위 디렉토리)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PDF_summary 폴더 경로
pdf_summary_path = Path(PROJECT_ROOT) / 'bill_summary' / 'PDF_summary'

# PDF_summary 폴더 내 csv 파일명에서 회차(숫자) 자동 추출
csv_files = [f for f in pdf_summary_path.glob('*.csv') if f.is_file()]
def extract_age(filename):
    match = re.search(r'(\d+)', filename.stem)
    return int(match.group(1)) if match else None

ages = sorted(set(extract_age(f) for f in csv_files if extract_age(f) is not None))

for age in ages:
    missing_path = os.path.join(PROJECT_ROOT, 'bill_summary', 'missing', f'bill_summary{age}_missing_titles.csv')
    pdf_path = os.path.join(PROJECT_ROOT, 'bill_summary', 'PDF_summary', f'{age}.csv')
    output_path = os.path.join(PROJECT_ROOT, 'billview', 'data', f'bill{age}.csv')

    if not (os.path.exists(missing_path) and os.path.exists(pdf_path)):
        print(f"❌ 파일 없음: {missing_path} 또는 {pdf_path}")
        continue

    print(f"▶ 처리 중: {age}회차")

    # 1. 누락 데이터 처리
    df = pd.read_csv(missing_path, encoding='utf-8-sig', dtype={'bill_number': str})

    def extract_bill_id(url):
        try:
            query = urlparse(url).query
            params = parse_qs(query)
            return params.get('billId', [''])[0]
        except:
            return ''

    def split_bill_title(text):
        if isinstance(text, str) and ']' in text:
            left, right = text.split(']', 1)
            bill_number = left.strip('[] ').strip()
            title = right.strip()
            return bill_number, title
        return '', text

    df['bill_id'] = df['url'].apply(extract_bill_id)
    df[['bill_number', 'title']] = df['bill_title'].apply(lambda x: pd.Series(split_bill_title(x)))
    df['bill_number'] = df['bill_number'].astype(str).str.strip()
    df['age'] = age

    # 2. PDF 요약 데이터 처리
    df_pdf = pd.read_csv(
        pdf_path, 
        encoding='utf-8-sig',
        dtype={'의안번호': str}
    )
    df_pdf = df_pdf.rename(columns={'의안번호': 'bill_number'})
    df_pdf['bill_number'] = df_pdf['bill_number'].str.strip()

    # 3. 의안번호 뒤 4자리 추출
    df['bill_last4'] = df['bill_number'].str[-4:].str.zfill(4)
    df_pdf['bill_last4'] = df_pdf['bill_number'].str[-4:].str.zfill(4)

    # 4. 병합 (뒤 4자리 기준)
    merged = pd.merge(
        df[['age', 'title', 'bill_id', 'bill_number', 'bill_last4']],
        df_pdf[['bill_last4', '제안이유', '주요내용or주 문']],
        on='bill_last4',
        how='left'
    )

    # 5. 최종 데이터 정리
    merged['제안이유'] = merged['제안이유'].fillna('').astype(str).str.strip()
    merged['주요내용or주 문'] = merged['주요내용or주 문'].fillna('').astype(str).str.strip()
    merged['content'] = (merged['제안이유'] + ' ' + merged['주요내용or주 문']).str.strip()

    final = merged[['age', 'title', 'bill_id', 'bill_number', '제안이유', '주요내용or주 문', 'content']]

    # 6. 저장
    final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'✅ 저장 완료: {output_path}')
    print(final.head())
