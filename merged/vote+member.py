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

# CSV 파일 불러오기     
df_members = pd.read_csv("~/OneDrive/바탕 화면/project/laws-radar/member_info/data/member_22.csv")       # 이름, 정당, 지역구 등
df_votes = pd.read_csv("~/OneDrive/바탕 화면/project/laws-radar/result_vote/data/vote_results_22.csv")         # 의안ID, 의안명, 의원명, 투표결과 등

# print(df_members.columns)
# print(df_votes.columns)

# 컬럼 공백 제거
df_votes.columns = df_votes.columns.str.strip()
df_members.columns = df_members.columns.str.strip()

# 병합: 의원 이름 기준 (컬럼명이 다를 경우 left_on / right_on 사용)
df_merged = pd.merge(
    df_votes,
    df_members,
    left_on="HG_NM",     # 투표결과 파일의 의원 이름 컬럼
    right_on="name",     # 기본정보 파일의 의원 이름 컬럼
    how="left"
)

# # 필요한 열만 추출
# df_selected = df_merged[[
#     "HG_NM",            # 의원 이름
#     # "POLY_NM",          # 정당
#     "RESULT_VOTE_MOD",  # 찬성/반대
#     "BILL_NAME",        # 의안 이름
#     "partyName",             # 정당 
#     "electoralDistrict",             #지역구
#     "gender"
# ]]
df_selected = df_merged.dropna(subset=["RESULT_VOTE_MOD", "BILL_NAME", "partyName", "gender"])



# # 결과 확인
# print(df_selected.head())

# BILL_NAME + 찬반별로 HG_NM, gender 등 분류
result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for _, row in df_selected.iterrows():
    bill = row["BILL_NAME"]
    vote = row["RESULT_VOTE_MOD"]
    
    result[bill][vote]["HG_NM"].append(row["HG_NM"])
    result[bill][vote]["gender"].append(row["gender"])
    result[bill][vote]["partyName"].append(row["partyName"])
    result[bill][vote]["electoralDistrict"].append(row["electoralDistrict"])

    # 1. 찬반 여부(RESULT_VOTE_MOD) 개수 세기
vote_counts = df_selected.groupby(["BILL_NAME", "RESULT_VOTE_MOD"]).size().unstack(fill_value=0)

# 2. 정당별 개수
party_counts = df_selected.groupby(["BILL_NAME", "partyName"]).size().unstack(fill_value=0)

# 3. 성별별 개수
gender_counts = df_selected.groupby(["BILL_NAME", "gender"]).size().unstack(fill_value=0)

# 4. 의안별 정렬 (찬성+반대 합 기준 정렬)
sorted_vote_counts = vote_counts.copy()
sorted_vote_counts["total"] = sorted_vote_counts.sum(axis=1)
sorted_vote_counts = sorted_vote_counts.sort_values(by="total", ascending=False)

# 의안명을 인덱스로 갖는 vote_counts, party_counts, gender_counts 병합
combined = vote_counts.copy()

# 병합 시 suffix로 구분
combined = combined.merge(party_counts, how="outer", left_index=True, right_index=True, suffixes=("", "_party"))
combined = combined.merge(gender_counts, how="outer", left_index=True, right_index=True, suffixes=("", "_gender"))

# NaN 값은 0으로 채움
combined = combined.fillna(0).astype(int)

# 저장 경로 지정
output_path = "C:/Users/1-16/OneDrive/바탕 화면/project/laws-radar/merged/data/combined_counts.csv"

# CSV로 저장
combined.to_csv(output_path, encoding="utf-8-sig")  # Excel 호환용
print(f"파일이 저장되었습니다: {output_path}")


with open(output_path, "w", encoding="utf-8") as f:
    for bill_name in sorted_vote_counts.index:
        f.write(f"\n[의안명] {bill_name}\n")

        f.write(" - 투표 결과:\n")
        if bill_name in vote_counts.index:
            f.write(str(vote_counts.loc[bill_name]) + "\n")
        else:
            f.write("데이터 없음\n")

        f.write(" - 정당 분포:\n")
        if bill_name in party_counts.index:
            f.write(str(party_counts.loc[bill_name]) + "\n")
        else:
            f.write("데이터 없음\n")

        f.write(" - 성별 분포:\n")
        if bill_name in gender_counts.index:
            f.write(str(gender_counts.loc[bill_name]) + "\n")
        else:
            f.write("데이터 없음\n")
        print("데이터 없음")


# # 예시 출력: 상위 3개 의안만 출력
# for bill_name, vote_data in list(result.items())[:3]:
#     print(f"\n[의안명] {bill_name}")
#     for vote_type, data in vote_data.items():
#         print(f"  - {vote_type}")
#         print(f"    * 의원: {data['HG_NM']}")
#         print(f"    * 성별: {data['gender']}")
#         print(f"    * 정당: {data['partyName']}")
#         print(f"    * 지역구: {data['electoralDistrict']}")

# (선택) 원래 병합 데이터 저장
# df_selected.to_csv("~/OneDrive/바탕 화면/project/laws-radar/merged/data/member_count1.csv", index=False)

# # 결과 확인
# print(df_selected.head())

# # 결과 저장
# df_selected.to_csv("~/OneDrive/바탕 화면/project/laws-radar/merged/data/vote+member.csv", index=False)


