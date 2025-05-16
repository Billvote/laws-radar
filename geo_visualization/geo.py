import pandas as pd
import os

# 공통 폴더 경로 변수화
base_dir = 'C:/Users/1-08/OneDrive/Desktop/DAMF2/Final_PJT/'

# -----------< 22대 >--------------
# 표결 결과 df 가져오기
vote_file_path = os.path.join(base_dir, 'result_vote/data/vote_results_22.csv')
voteDf_22 = pd.read_csv(vote_file_path)

# 의원 정보 df 가져오기
member_file_path = os.path.join(base_dir, 'member_info/data/member_22.csv')
memberDf_22 = pd.read_csv(member_file_path)

# 각 df에서 필요 컬럼만 선택
vote_selected = ['HG_NM', 'BILL_NAME', 'VOTE_DATE', 'RESULT_VOTE_MOD']
voteDf_22 = voteDf_22[vote_selected]

memberDf_22 = memberDf_22.drop('committees', axis=1)

#### 화나는 부분;;;;;;
# merge: name, HG_NM 기준 병합

# 표결에 멤버 병합
# merged_df = voteDf_22.merge(memberDf_22, left_on='HG_NM', right_on='name', how='left')
# merged_df = merged_df.drop('name', axis=1)

# 멤버 명단에 표결 병합: 가결안에 투표 안 한 사람 41명
merged_df = memberDf_22.merge(voteDf_22, left_on='name', right_on='HG_NM', how='left')
print(merged_df[merged_df['HG_NM'].isna()])


# -------의안에 따른, 당별 표결: 통계 분석-----------------
# 현황 테이블
vote_counts = merged_df.groupby(['BILL_NAME', 'partyName'])['RESULT_VOTE_MOD'].value_counts().unstack(fill_value=0)

# 당별 표결 비율 계산
vote_ratios = vote_counts.div(vote_counts.sum(axis=1), axis=0)

# print(vote_ratios.sort_values(by='반대', ascending=False))
# print(vote_counts.sort_values(by='불참', ascending=False))

# print(vote_ratios.head(20))

# 무엇을 더 분석할지 고민 필요: 성별 따른 분포.. 의안 따른 분포.. 기권을 볼지, 반대를 볼지 등



# -----merged_df 갖고 geojson 적용 해야 함------