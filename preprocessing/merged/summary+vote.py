import pandas as pd
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 1) 표결 결과 불러오기
vote_path = settings.BASE_DIR / 'result_vote' / 'data' / 'vote_results_20.csv'
voteDf = pd.read_csv(vote_path)

# 필요한 컬럼만 선택 + 중복 제거 (법안당 1개)
voteDf = voteDf[['BILL_NAME', 'BILL_NO', 'BILL_ID', 'BILL_URL']]
voteDf['BILL_URL'] = voteDf['BILL_URL'].str.strip()
vote_unique = voteDf.drop_duplicates(subset='BILL_URL')

print(f"vote 데이터 행 수 (전체): {len(voteDf)}")
print(f"vote 데이터 행 수 (중복 제거): {len(vote_unique)}")

# 2) summary 불러오기
summary_path = settings.BASE_DIR / 'bill_summary' / 'data' / 'bill_summary20.csv'
summaryDf = pd.read_csv(summary_path, on_bad_lines='skip', encoding='utf-8')
summaryDf['url'] = summaryDf['url'].str.strip()

print(f"📄 summary 데이터 행 수: {len(summaryDf)}")

# 3) 병합
merged_df = pd.merge(
    vote_unique,
    summaryDf,
    left_on='BILL_URL',
    right_on='url',
    how='inner'
)

print(f"병합된 행 수: {len(merged_df)}")

merged_df = merged_df[['BILL_NAME', 'BILL_NO', 'BILL_ID', 'summary']]
print(merged_df.head())

# 4) 병합 결과 저장
output_path = settings.BASE_DIR / 'merged' / 'data' / 'summary_vote_20.csv'
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n병합된 데이터 저장 완료: {output_path}")
