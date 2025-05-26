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

# # 대수별 CSV 파일 경로
# paths = {
#     20: settings.BASE_DIR / 'merged' / 'data' / 'summary_vote_20.csv',
#     21: settings.BASE_DIR / 'merged' / 'data' / 'summary_vote_21.csv',
#     22: settings.BASE_DIR / 'merged' / 'data' / 'summary_vote_22.csv',
# }

# # 파일별 age 컬럼 추가
# dfs = []
# for age, path in paths.items():
#     df = pd.read_csv(path)
#     df['age'] = age
#     dfs.append(df)

# # 병합
# merged_df = pd.concat(dfs, ignore_index=True)
# merged_df = merged_df[['age', 'BILL_NAME', 'BILL_ID', 'BILL_NO', 'summary']].rename(columns={
#     'BILL_NAME': 'title',
#     'BILL_ID': 'bill_id',
#     'BILL_NO': 'bill_number',
#     'summary': 'content'
#     })
# print(merged_df.head())
# print(merged_df['age'].value_counts())

# --------------------------------------------------
# 지역구: age, SGG_code, SIDO_SGG, SIDO, SGG, boundary

# 대수별 CSV 파일 경로
# paths = {
#     20: settings.BASE_DIR / 'member_info' / 'data' / 'member_20.csv',
#     21: settings.BASE_DIR / 'member_info' / 'data' / 'member_21.csv',
#     22: settings.BASE_DIR / 'member_info' / 'data' / 'member_22.csv'
# }

# # 파일별 age 컬럼 추가
# dfs = []
# for age, path in paths.items():
#     df = pd.read_csv(path)
#     df['age'] = age
#     dfs.append(df)

# # 병합
# merged_df = pd.concat(dfs, ignore_index=True)
# # filtered_df = merged_df[~merged_df['electoralDistrict'].str.contains('비례대표', na=False)]

# # 먼저 컬럼 생성 (기본값: None)
# merged_df['sido'] = None
# merged_df['sgg'] = None

# # 비례대표가 아닌 경우만 분리
# mask = ~merged_df['electoralDistrict'].str.contains('비례대표', na=False)

# # 지역구인 경우만 분할
# split_df = merged_df.loc[mask, 'electoralDistrict'].str.split(' ', n=1, expand=True)

# # split 결과가 2개 컬럼인 경우만 할당 (즉, 띄어쓰기가 제대로 있는 경우만)
# valid_split = split_df[1].notna()

# # sido, sgg에 각각 값 넣기
# merged_df.loc[mask & valid_split, 'sido'] = split_df.loc[valid_split, 0]
# merged_df.loc[mask & valid_split, 'sgg'] = split_df.loc[valid_split, 1]

# # merged_df[['sido', 'sgg']] = merged_df['electoralDistrict'].str.split(' ', n=1, expand=True)
# merged_df = merged_df[['age', 'electoralDistrict', 'sido', 'sgg']].rename(columns={'electoralDistrict': 'sido_sgg'})

# # '세종특별자치시'가 포함된 행에 대해 sido와 sgg 모두 지정
# mask_sejong = merged_df['sido_sgg'].str.contains('세종특별자치시', na=False)
# merged_df.loc[mask_sejong, 'sido'] = '세종'
# merged_df.loc[mask_sejong, 'sgg'] = merged_df.loc[mask_sejong, 'sido_sgg']

# # print(merged_df.loc[mask].isna().value_counts())
# print(merged_df.tail())
# ---------------------------------------------------

# district_22.csv에 geojson 내용 파싱해 붙여넣기

# # --------------------------------------------------
# # 저장
# output_path = settings.BASE_DIR / 'geovote' / 'data' / 'district.csv'
# merged_df.to_csv(output_path, index=False, na_rep='NULL')
# print(f"✅ CSV 저장 완료: {output_path}")