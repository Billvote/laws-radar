import pandas as pd
import os

# 기본 경로
base_dir = 'C:/Users/1-08/OneDrive/Desktop/DAMF2/Final_PJT/'

# 1) 표결 결과 불러오기
vote_file_path = os.path.join(base_dir, 'result_vote/data/vote_results_22.csv')
voteDf_22 = pd.read_csv(vote_file_path)

# 필요한 컬럼만 선택 + 중복 제거 (법안당 1개)
voteDf_22 = voteDf_22[['BILL_NAME', 'BILL_NO', 'BILL_URL']]
voteDf_22['BILL_URL'] = voteDf_22['BILL_URL'].str.strip()
vote_unique = voteDf_22.drop_duplicates(subset='BILL_URL')

print(f"vote 데이터 행 수 (전체): {len(voteDf_22)}")
print(f"vote 데이터 행 수 (중복 제거): {len(vote_unique)}")

# 2) summary 불러오기
summary_file_path = os.path.join(base_dir, 'bill_summary/data/bill_summary22.csv')
summaryDf_22 = pd.read_csv(summary_file_path, on_bad_lines='skip', encoding='utf-8')
summaryDf_22['url'] = summaryDf_22['url'].str.strip()

print(f"📄 summary 데이터 행 수: {len(summaryDf_22)}")

# 3) 병합
merged_df = pd.merge(
    vote_unique,
    summaryDf_22,
    left_on='BILL_URL',
    right_on='url',
    how='inner'
)

print(f"병합된 행 수: {len(merged_df)}")

merged_df = merged_df[['BILL_NAME', 'BILL_NO', 'summary']]
print(merged_df.head())

# 4) 병합 결과 저장
output_path = os.path.join(base_dir, 'merged/data/summary+vote_22.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 폴더 없으면 생성
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n병합된 데이터 저장 완료: {output_path}")
