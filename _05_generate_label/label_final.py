# 라벨링 작업 하는 코드
# merge.py 사용 후 제일 마지막에 사용하는 코드

import pandas as pd
import re
from pathlib import Path

# 현재 파일 기준으로 상대경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent  # laws-radar 기준
DATA_DIR = BASE_DIR / "geovote" / "data"
file_path = DATA_DIR / "merged_bill_data.csv"
output_path = DATA_DIR / "merged_bill_data_labeled.csv"

# CSV 파일 로드 (인코딩 자동 처리)
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

# title 정규화 함수
def add_label_column(df: pd.DataFrame) -> pd.DataFrame:
    def normalize_title(title):
        if pd.isna(title):
            return ''

        title = re.sub(r'\([^)]*\)', '', title) # 괄호 및 괄호 안 내용 제거
        title = re.sub(r'[^\w\s]', '', title) # 특수문자 제거
        title = re.sub(r'\d+차', '차', title) # 숫자+차 → 차로 통일 (예: 1차, 2차 등)
        title = re.sub(r'\s+', ' ', title).strip() # 공백 정규화
        return title

    # cluster_keyword 컬럼에서 대괄호 제거
    # df['cluster_keyword'] = df['cluster_keyword'].astype(str).str.strip('[]')

    # 정규화된 title로 라벨 생성
    df['label'] = pd.factorize(df['title'].apply(normalize_title))[0]
    return df

# 저장
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"라벨링된 파일이 저장되었습니다: {output_path}")
