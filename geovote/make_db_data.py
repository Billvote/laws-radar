import pandas as pd
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 1. 정당: 대수, 정당 이름
# all_party_path = settings.MEMBER_INFO_DATA_DIR / 'all_member_18-22.csv'
# df = pd.read_csv(all_party_path, encoding='utf-8-sig')
# # print(df.columns)

# df = df[['대수', '정당']].rename(columns={'대수': 'age', '정당': 'party'})
# filtered = df[df['age'].isin([20, 21, 22])]
# unique_parties = filtered['party'].drop_duplicates().reset_index(drop=True)

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

# < 의원 테이블: age, name, party, sido_sgg, member_id, gender 값 필요

# paths = {
#     20: settings.BASE_DIR / 'member_info' / 'data' / 'members20.csv',
#     21: settings.BASE_DIR / 'member_info' / 'data' / 'members21.csv',
#     22: settings.BASE_DIR / 'member_info' / 'data' / 'members22.csv'
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

# merged_df = merged_df.rename(columns={
#     'monaCode': 'member_id',
#     '이름': 'name',
#     '정당': 'party',
#     '선거구': 'SIDO_SGG',
#     '성별': 'gender'
#     })
# merged_df = merged_df[['age', 'name', 'party', 'SIDO_SGG', 'member_id', 'gender']] # 순서 정렬

# print(merged_df.head())

# 표결 --------------------------------------------------
# 모델링 수정: 의원id, 의안 번호나 id, 정당, 성별 있어야 할 듯, 찬성반대기권불참 

# vote 병합
# paths = {
#     20: settings.BASE_DIR / 'result_vote' / 'data' / 'vote_20.csv',
#     21: settings.BASE_DIR / 'result_vote' / 'data' / 'vote_21.csv',
#     22: settings.BASE_DIR / 'result_vote' / 'data' / 'vote_22.csv'
#     }

# dfs = []
# for age, path in paths.items():
#     df = pd.read_csv(path)
#     dfs.append(df)

# vote_df = pd.concat(dfs, ignore_index=True)
# vote_df = vote_df[['AGE', 'MONA_CD', 'BILL_ID', 'RESULT_VOTE_MOD']].rename(
#     columns={
#         'AGE': 'age',
#         'MONA_CD': 'member_id',
#         'BILL_ID': 'bill_id',
#         'RESULT_VOTE_MOD': 'vote_result',
#     })

# bill_number 추가하기..
# vote_path = settings.BASE_DIR / 'geovote' / 'data' / 'vote.csv'
# bill_path = settings.BASE_DIR / 'geovote' / 'data' / 'bill.csv'

# vote_df = pd.read_csv(vote_path)
# bill_df = pd.read_csv(bill_path)
# bill_df['bill_number'] = bill_df['bill_number'].astype(str)

# merged_df = vote_df.merge(bill_df[['bill_id', 'bill_number']], on='bill_id', how='left')

# 기존 CSV 불러오기
path = settings.BASE_DIR / 'geovote' / 'data' / 'vote.csv'

df = pd.read_csv(path)

# bill_id 컬럼 삭제
df = df.drop(columns=['bill_id'])
df['bill_number'] = df['bill_number'].fillna(0).astype(int).astype(str)

# 새로운 CSV 저장 (bill_id 제외)
# df.to_csv('수정된파일.csv', index=False)

# print(monaCd_df.head())

# 저장
output_path = settings.BASE_DIR / 'geovote' / 'data' / 'vote(1).csv'
df.to_csv(output_path, index=False, na_rep='NULL')
print(f"✅ CSV 저장 완료: {output_path}")