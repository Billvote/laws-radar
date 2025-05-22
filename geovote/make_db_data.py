import pandas as pd
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 1. 정당: 대수, 정당 이름
# all_party_path = settings.MEMBER_INFO_DATA_DIR / 'all_member_18-22.csv'
# df = pd.read_csv(all_party_path)

# df = df[['대수', '정당']].rename(columns={'대수': 'age', '정당': 'party'})
# filtered = df[df['age'].isin([20, 21, 22])]
# unique_parties = filtered.drop_duplicates(subset=['age', 'party']).reset_index(drop=True)

# ------------------------------------------------------------
# 2. bill: age, title, bill_id, bill_number, content

# 1. 각 대수별 CSV 파일 경로
paths = {
    20: settings.BASE_DIR / 'merged' / 'data' / 'summary_vote_20.csv',
    21: settings.BASE_DIR / 'merged' / 'data' / 'summary_vote_21.csv',
    22: settings.BASE_DIR / 'merged' / 'data' / 'summary_vote_22.csv',
}

# 2. DataFrame 리스트에 각 파일을 읽고 age 컬럼 추가
dfs = []
for age, path in paths.items():
    df = pd.read_csv(path)
    df['age'] = age
    dfs.append(df)

# 3. 모두 병합
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df[['age', 'BILL_NAME', 'BILL_ID', 'BILL_NO', 'summary']].rename(columns={
    'BILL_NAME': 'title',
    'BILL_ID': 'bill_id',
    'BILL_NO': 'bill_number',
    'summary': 'content'
    })

# 4. 결과 확인
print(merged_df.head())
print(merged_df['age'].value_counts())

# --------------------------------------------------
# 저장
output_path = settings.BASE_DIR / 'geovote' / 'data' / 'bill.csv'
merged_df.to_csv(output_path, index=False)
print(f"✅ CSV 저장 완료: {output_path}")