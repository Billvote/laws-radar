# import pandas as pd
# import os

# # 공통 폴더 경로 변수화
# base_dir = 'C:/Users/1-08/OneDrive/Desktop/DAMF2/Final_PJT/'

# # -----------< 22대 >--------------
# # 표결 결과 df 가져오기
# vote_file_path = os.path.join(base_dir, 'result_vote/data/vote_results_22.csv')
# voteDf_22 = pd.read_csv(vote_file_path)

# # 의원 정보 df 가져오기
# member_file_path = os.path.join(base_dir, 'member_info/data/member_22.csv')
# memberDf_22 = pd.read_csv(member_file_path)

# # 각 df에서 필요 컬럼만 선택
# vote_selected = ['HG_NM', 'BILL_NAME', 'VOTE_DATE', 'RESULT_VOTE_MOD']
# voteDf_22 = voteDf_22[vote_selected]

# memberDf_22 = memberDf_22.drop('committees', axis=1)

# #### 화나는 부분;;;;;;
# # merge: name, HG_NM 기준 병합

# # 표결에 멤버 병합
# # merged_df = voteDf_22.merge(memberDf_22, left_on='HG_NM', right_on='name', how='left')
# # merged_df = merged_df.drop('name', axis=1)

# # 멤버 명단에 표결 병합: 가결안에 투표 안 한 사람 41명
# merged_df = memberDf_22.merge(voteDf_22, left_on='name', right_on='HG_NM', how='left')
# print(merged_df[merged_df['HG_NM'].isna()])


# # -------의안에 따른, 당별 표결: 통계 분석-----------------
# # 현황 테이블
# vote_counts = merged_df.groupby(['BILL_NAME', 'partyName'])['RESULT_VOTE_MOD'].value_counts().unstack(fill_value=0)

# # 당별 표결 비율 계산
# vote_ratios = vote_counts.div(vote_counts.sum(axis=1), axis=0)

# # print(vote_ratios.sort_values(by='반대', ascending=False))
# # print(vote_counts.sort_values(by='불참', ascending=False))

# # print(vote_ratios.head(20))

# # 무엇을 더 분석할지 고민 필요: 성별 따른 분포.. 의안 따른 분포.. 기권을 볼지, 반대를 볼지 등



# # -----merged_df 갖고 geojson 적용 해야 함------


#==================================================================================
import pandas as pd
from collections import defaultdict
from pathlib import Path
from collections import Counter

# 홈 디렉토리를 기준으로 경로 지정
base_dir = Path.home() /"Desktop" / "project" / "laws-radar"

# 파일 경로 정의
members_path = base_dir / "member_info" / "data" / "member_22.csv"
votes_path = base_dir / "result_vote" / "data" / "vote_results_22.csv"

# CSV 파일 불러오기
df_members = pd.read_csv(members_path)
df_votes = pd.read_csv(votes_path)

print(df_members.columns)
print(df_votes.columns)

# 확인
# print(df_members.head())

# # 컬럼 공백 제거
# df_votes.columns = df_votes.columns.str.strip()
# df_members.columns = df_members.columns.str.strip()

df_merged = pd.merge(
    df_votes,
    df_members,
    left_on='HG_NM',
    right_on='name',
    how='left'
)


# 필요한 열만 추출
df_selected = df_merged[[
    "HG_NM",            # 의원 이름
    "RESULT_VOTE_MOD",  # 찬성/반대
    "BILL_NAME",        # 의안 이름
    "partyName",             # 정당 
    "electoralDistrict",             #지역구
    "gender"
]]
df_selected = df_selected.dropna(subset=["RESULT_VOTE_MOD", "BILL_NAME", "partyName", "gender"])

df_summary = df_selected.groupby(
    ["BILL_NAME", "RESULT_VOTE_MOD"]
).agg(
    members = ("HG_NM", list),
    parties = ("partyName", list),
    genders = ("gender", list),
    districts = ("electoralDistrict", list),
).reset_index()


def count_list(x):
    return dict(Counter(x))

df_summary_counts = df_selected.groupby(
    ["BILL_NAME", "RESULT_VOTE_MOD"]
).agg(
    members = ("HG_NM", count_list),
    parties = ("partyName", count_list),
    genders = ("gender", count_list),
    districts = ("electoralDistrict", count_list),
).reset_index()

print(df_summary_counts.head())

df_summary_counts.to_csv("summary_counts.csv", index=False, encoding="utf-8-sig")


# columns_to_select = ["HG_NM", "BILL_NAME", "RESULT_VOTE_MOD", "gender", "electoralDistrict", "partyName"]
# df_lawmaker_votes = df_selected[columns_to_select]

# # 확인
# print(df_lawmaker_votes.head())

# (선택) 원래 병합 데이터 저장
# df_selected.to_csv("~/OneDrive/바탕 화면/project/laws-radar/merged/data/member_count1.csv", index=False)

# # 결과 확인
# print(df_selected.head())

# # 결과 저장
# df_selected.to_csv("~/OneDrive/바탕 화면/project/laws-radar/merged/data/vote+member.csv", index=False)


