import pandas as pd
import re

# 파일 경로
file_path = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\merged_bill_data.csv"
output_path = file_path.replace('.csv', '_labeled.csv')

# CSV 파일 로드 (인코딩 자동 처리)
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

# title 정규화 함수
def normalize_title(title):
    if pd.isna(title):
        return ''
    # 괄호 및 괄호 안 내용 제거
    title = re.sub(r'\([^)]*\)', '', title)
    # 특수문자 제거
    title = re.sub(r'[^\w\s]', '', title)
    # 숫자+차 → 차로 통일 (예: 1차, 2차 등)
    title = re.sub(r'\d+차', '차', title)
    # 공백 정규화
    title = re.sub(r'\s+', ' ', title).strip()
    return title

# 정규화된 title로 라벨 생성
df['label'] = pd.factorize(df['title'].apply(normalize_title))[0]

# 저장
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"라벨링된 파일이 저장되었습니다: {output_path}")
