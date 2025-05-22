
# import requests
# import pandas as pd

# all_members = []
# page = 1
# limit = 100  # 한 번에 최대한 많이 가져오기 (API가 허용하는 범위 내)

# while True:
#     url = f"https://openwatch.kr/api/national-assembly/members?page={page}&limit={limit}"
#     response = requests.get(url, headers={"Accept": "*/*"})
#     data = response.json()
    
#     members = data["rows"]
#     if not members:
#         break  # 더 이상 데이터 없으면 반복 종료
    
#     all_members.extend(members)
    
#     total = data["pagination"]["totalCount"]
#     if len(all_members) >= total:
#         break  # 모든 데이터 다 받았으면 종료
    
#     page += 1

# # 모든 페이지에서 모은 데이터를 DataFrame으로 변환
# df = pd.DataFrame(all_members)

# # 원하는 컬럼만 선택 (컬럼명은 API 데이터에 맞게 조정)
# selected_columns = df[["name", "partyName", "electoralDistrict", "gender", "committees"]]

# # CSV로 저장
# selected_columns.to_csv("data/member_info.csv", index=False, encoding="utf-8-sig")

# print(f"총 {len(df)}명의 의원 데이터를 저장했습니다.")

# --------------------------------------20, 21, 22대 국회의원들만 추출--------------------------------------------------------------------
import pandas as pd

# 원본 CSV 파일 불러오기
df = pd.read_csv("data/all_member_18-22.csv")  # 파일명을 실제로 존재하는 파일명으로 바꿔주세요

# 저장할 대수 목록
target_assemblies = [20, 21, 22]

# 대수별로 필터링 후 저장
for assembly_num in target_assemblies:
    group_df = df[df["대수"] == assembly_num]
    
    # 저장할 컬럼 선택 (원하시면 전체 컬럼 그대로 저장해도 됩니다)
    selected_cols = group_df[["이름", "정당", "선거구", "성별"]]  # 필요에 따라 수정
    
    # 저장
    filename = f"data/members_assembly_{assembly_num}.csv"
    selected_cols.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"{assembly_num}대 국회의원 {len(group_df)}명 데이터를 {filename}에 저장했습니다.")

#----------------------이상한 출력이 나오는 ...-------------------------------------------------------

# import requests
# import pandas as pd

# all_members = []
# page = 1
# limit = 100

# while True:
#     url = f"https://openwatch.kr/api/national-assembly/members?page={page}&limit={limit}"
#     response = requests.get(url, headers={"Accept": "*/*"})
#     data = response.json()
    
#     members = data["rows"]
#     if not members:
#         break
    
#     all_members.extend(members)
    
#     total = data["pagination"]["totalCount"]
#     if len(all_members) >= total:
#         break
    
#     page += 1

# df = pd.DataFrame(all_members)


# # 20대 ~ 22대 의원만 필터링
# df_filtered = df[df["ages"].isin([20, 21, 22])]

# # 대수별로 각각 저장
# # 20대 ~ 22대 의원만 필터링 (ages 리스트에 20,21,22 중 하나라도 포함된 경우)
# target_assemblies = {20, 21, 22}
# df_filtered = df[df["ages"].apply(lambda x: any(a in target_assemblies for a in x))]

# # 대수별로 각각 저장
# for assembly_num in target_assemblies:
#     group_df = df_filtered[df_filtered["ages"].apply(lambda x: assembly_num in x)]
#     filename = f"data/members_assembly_{assembly_num}.csv"
#     selected_cols = group_df[["name", "partyName", "electoralDistrict", "gender", "committees"]]
#     selected_cols.to_csv(filename, index=False, encoding="utf-8-sig")
#     print(f"{assembly_num}대 국회의원 {len(group_df)}명 데이터를 {filename}에 저장했습니다.")




# =========================================칼럼 참고자료 =================================================================
# {
#   "pagination": {
#     "totalCount": 45
#   },
#   "rows": [
#     {
#       "id": "14M56632",
#       "hjId": 2800,
#       "name": "강기윤",
#       "nameHanja": "姜起潤",
#       "nameEnglish": "KANG GIYUN",
#       "birthdateType": "음",
#       "birthdate": "1960-06-04",
#       "gender": "남",
#       "reelected": "재선",
#       "partyName": "새누리당",
#       "job": "간사",
#       "tel": "02-784-1751",
#       "committees": "보건복지위원회",
#       "email": "ggotop@naver.com",
#       "homepage": "http://blog.naver.com/ggotop",
#       "staff": "김홍광,한영애",
#       "secretary": "김샛별,장원종",
#       "secretary2": "안효상,홍지형,이유진,김지훈,조옥자",
#       "profile": "[학력]\n마산공고(26회)\n창원대학교\n행정학과\n중앙대학교 행정대학원\n지방의회과 석사\n창원대학교 대학원\n행정학 박사\n[경력]\n현) 국회\n보건복지위원회 국민의힘 간사\n현)\n국민의힘 소상공인살리기 특별위원회 부위원장\n현) 국민의힘 코로나19 대책 특별위원회 위원\n미래통합당 경남도당 민생특위 위원장\n제19대 국회의원 (새누리당/경남 창원시 성산구)\n새누리당 원내부대표",
#       "officeAddress": "의원회관 937호",
#       "electoralDistrict": "경남 창원시성산구",
#       "electoralDistrictType": "지역구"
#     }
#   ]
# }

