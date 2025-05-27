# 결측의 결측 필터링

import pandas as pd

# CSV 파일 읽기
df = pd.read_csv(r'C:\Users\1-02\Desktop\DAMF2\laws-radar\billview\merged_output\missing_content.csv')

# title 컬럼에서 제거할 키워드들을 포함하는 행 찾기
# 정규표현식을 사용하여 여러 키워드를 한번에 검색
keywords_pattern = '예산안|변경안|회계'
mask = df['title'].str.contains(keywords_pattern, na=False)

# 제거될 행들(키워드 포함된 행들) 추출
removed_rows = df[mask]

# 원본 데이터에서 키워드 포함된 행들 제거 (키워드가 없는 행들만 남김)
filtered_df = df[~mask]

# 결과 확인
print(f"원본 데이터 행 수: {len(df)}")
print(f"제거된 행 수: {len(removed_rows)}")
print(f"필터링 후 행 수: {len(filtered_df)}")

# 결과를 각각 CSV 파일로 저장
filtered_df.to_csv('filtered_data.csv', index=False, encoding='utf-8-sig')
removed_rows.to_csv('removed_data.csv', index=False, encoding='utf-8-sig')

print("필터링 완료!")
print("- 필터링된 데이터: filtered_data.csv")
print("- 제거된 데이터: removed_data.csv")
