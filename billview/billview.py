import os
import pandas as pd
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import re

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PDF 요약 데이터 경로
pdf_summary_path = Path(PROJECT_ROOT) / 'bill_summary' / 'PDF_summary'

# 회차 추출 함수
def extract_age(filename):
    match = re.search(r'(\d+)', filename.stem)
    return int(match.group(1)) if match else None

# 처리 가능한 회차 목록 생성
csv_files = [f for f in pdf_summary_path.glob('*.csv') if f.is_file()]
ages = sorted(set(extract_age(f) for f in csv_files if extract_age(f) is not None))

# 주요 데이터 처리 루프
for age in ages:
    # 파일 경로 설정
    missing_path = os.path.join(PROJECT_ROOT, 'bill_summary', 'missing', f'bill_summary{age}_missing_titles.csv')
    pdf_path = os.path.join(PROJECT_ROOT, 'bill_summary', 'PDF_summary', f'{age}.csv')
    output_path = os.path.join(PROJECT_ROOT, 'billview', 'data', f'bill{age}.csv')

    # 파일 존재 여부 확인
    if not (os.path.exists(missing_path) and os.path.exists(pdf_path)):
        print(f"❌ 파일 없음: {missing_path} 또는 {pdf_path}")
        continue

    print(f"\n▶ {age}회차 데이터 처리 시작")

    # 1. 누락 데이터 로드 및 전처리
    df_missing = pd.read_csv(
        missing_path,
        encoding='utf-8-sig',
        dtype={'bill_number': str}
    )
    
    # URL 파싱을 통한 bill_id 추출
    df_missing['bill_id'] = df_missing['url'].apply(
        lambda x: parse_qs(urlparse(x).query).get('billId', [''])[0]
    )
    
    # 법안 제목 분할 처리
    def split_title(text):
        if isinstance(text, str) and ']' in text:
            parts = text.split(']', 1)
            return parts[0].strip('[] '), parts[1].strip()
        return '', text
    
    df_missing[['bill_number', 'title']] = df_missing['bill_title'].apply(
        lambda x: pd.Series(split_title(x))
    )
    df_missing['age'] = age

    # 2. PDF 요약 데이터 로드
    df_pdf = pd.read_csv(
        pdf_path,
        encoding='utf-8-sig',
        dtype={'의안번호': str}
    ).rename(columns={'의안번호': 'bill_number'})
    
    # 데이터 정제
    df_pdf['bill_number'] = df_pdf['bill_number'].str.strip()

    # 3. 키 생성: 의안번호 뒤 4자리
    for df in [df_missing, df_pdf]:
        df['bill_last4'] = df['bill_number'].str[-4:].str.zfill(4)

    # 4. 데이터 병합
    merged = pd.merge(
        df_missing[['age', 'title', 'bill_id', 'bill_number', 'bill_last4']],
        df_pdf[['bill_last4', '제안이유', '주요내용or주 문']],
        on='bill_last4',
        how='left'
    )

    # 5. 컨텐츠 생성 및 필터링 (공백만 있는 경우도 완전히 제외)
    merged['content'] = (
        merged['제안이유'].fillna('') + ' ' + 
        merged['주요내용or주 문'].fillna('')
    ).str.strip()
    
    # content가 빈 문자열이거나 공백만 있는 경우 모두 제외
    filtered = merged[merged['content'].str.strip() != '']

    # 6. 최종 출력 형식
    final_output = filtered[[
        'age', 
        'title', 
        'bill_id', 
        'bill_number', 
        'content'
    ]]

    # 7. 결과 저장
    final_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ {age}회차 처리 완료: {len(final_output)}건 저장")
    print("샘플 데이터:")
    print(final_output.head(3).to_markdown(index=False))
