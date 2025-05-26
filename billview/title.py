import pandas as pd
from urllib.parse import urlparse, parse_qs
from pathlib import Path  # 추가된 부분

# 파일 경로 설정
missing_path = r'C:\Users\1-02\Desktop\DAMF2\laws-radar\bill_summary\missing\bill_summary22_missing_titles.csv'
pdf_path = r'C:\Users\1-02\Desktop\DAMF2\laws-radar\bill_summary\PDF_summary\22.csv'
output_path = r'C:\Users\1-02\Desktop\DAMF2\laws-radar\billview\data\bill22.csv'

# 1. 누락 데이터 처리 (문자열로 읽기)
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

# PDF 파일명에서 age 추출 (변경된 부분)
pdf_age = int(Path(pdf_path).stem)  # 파일명에서 확장자 제거 후 정수 변환
df['age'] = pdf_age  # 기존 하드코딩 22 대체

# 2. PDF 요약 데이터 처리 (의안번호 문자열로 강제 변환)
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
