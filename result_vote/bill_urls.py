# bill url 추출 코드

import pandas as pd

import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

paths = {
    20: settings.BASE_DIR / 'result_vote' / 'data' / 'vote_results_20.csv',
    21: settings.BASE_DIR / 'result_vote' / 'data' / 'vote_results_21.csv',
    22: settings.BASE_DIR / 'result_vote' / 'data' / 'vote_results_22.csv'
}

# vote_results 파일에서 url 추출
dfs = []
for age, path in paths.items():
    df = pd.read_csv(path)
    dfs.append(df)
all_bill_df = pd.concat(dfs, ignore_index=True)

all_bill_df = all_bill_df[['BILL_ID', 'BILL_URL']]
unique_df = all_bill_df.drop_duplicates(subset='BILL_ID').rename(
    columns={
        # 'BILL_NO': 'bill_number',
        # 'BILL_NAME': 'title',
        'BILL_ID': 'bill_id',
        'BILL_URL': 'url'
    }
)

# 최종 csv로 만들기
file_path = settings.BASE_DIR / 'geovote' / 'data' / 'merged_bill_data_labeled.csv'
df = pd.read_csv(file_path)

merged_df = pd.merge(unique_df, df, how='inner', on='bill_id')

output_path = settings.BASE_DIR / 'result_vote' / 'data' / 'urls.csv'
merged_df.to_csv(output_path, index=False, na_rep='NULL')

# print(merged_df.info())
# print(merged_df.columns)