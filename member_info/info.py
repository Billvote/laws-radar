# --------------------------------------20, 21, 22대 국회의원들만 추출--------------------------------------------------------------------
import pandas as pd

# 원본 CSV 파일 불러오기
df = pd.read_csv("data/all_member_18-22.csv") 

# 저장할 대수 목록
target_assemblies = [20, 21, 22]

# 대수별로 필터링 후 저장
for assembly_num in target_assemblies:
    group_df = df[df["대수"] == assembly_num]
    
    # 저장할 컬럼 선택 (원하시면 전체 컬럼 그대로 저장해도 됩니다)
    selected_cols = group_df[["monaCode", "이름", "정당", "선거구", "성별"]]  # 필요에 따라 수정
    
    # 저장
    filename = f"data/members_{assembly_num}.csv"
    selected_cols.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"{assembly_num}대 국회의원 {len(group_df)}명 데이터를 {filename}에 저장했습니다.")