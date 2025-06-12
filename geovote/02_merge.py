import pandas as pd
import re

# 파일 경로 설정
base_file_path = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\bill_filtered_final.csv"
cluster_file_path = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\keword_clustering\data\bill_keyword_clustering.csv"
summary_file_path = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\summary_of_content.csv"
output_path = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\merged_bill_data.csv"

# 1. 파일 읽기
base_df = pd.read_csv(base_file_path, encoding='utf-8', on_bad_lines='warn')
cluster_df = pd.read_csv(cluster_file_path, encoding='utf-8', on_bad_lines='warn')
summary_df = pd.read_csv(summary_file_path, encoding='utf-8', engine='python', on_bad_lines='warn')

# 2. 컬럼명 변환
cluster_df = cluster_df.rename(columns={'topic': 'cluster', 'topic_label': 'cluster_keyword'})
summary_df = summary_df.rename(columns={'content': 'summary'})  # content → summary

# 3. 필요한 컬럼만 추출
base_columns = ['age', 'title', 'bill_id', 'bill_number']
cluster_columns = ['bill_id', 'cluster', 'cluster_keyword', 'content']  # content 컬럼 추가
summary_columns = ['bill_id', 'summary']

base_df = base_df[base_columns]
cluster_df = cluster_df[cluster_columns]
summary_df = summary_df[summary_columns]

# 4. content 컬럼을 cleaned로 변경
cluster_df = cluster_df.rename(columns={'content': 'cleaned'})

# 5. 데이터 병합
merged_df = pd.merge(
    base_df,
    cluster_df,
    on='bill_id',
    how='left'
)
merged_df = pd.merge(
    merged_df,
    summary_df,
    on='bill_id',
    how='left'
)

# 6. cluster_keyword에서 숫자와 쌍따옴표, 괄호만 제거하고 키워드만 남기기
def format_keywords(x):
    if pd.isna(x): 
        return '[]'
    # 괄호 제거
    cleaned = str(x).replace('(', ' ').replace(')', ' ')
    # 맨 앞에 숫자(1개 이상)와 공백 제거
    cleaned = re.sub(r'^\s*\d+\s*', '', cleaned)
    # 쌍따옴표 제거
    cleaned = cleaned.replace('"', '').strip()
    # 중복 공백 제거
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # 앞뒤 쉼표, 공백 정리
    cleaned = cleaned.strip(' ,')
    # 빈 값 처리
    return f'[{cleaned}]' if cleaned else '[]'

merged_df['cluster_keyword'] = merged_df['cluster_keyword'].apply(format_keywords)

# 7. 최종 컬럼 순서 지정 (cleaned 컬럼 추가)
final_columns = ['age', 'title', 'bill_id', 'bill_number', 'cleaned', 'summary', 'cluster', 'cluster_keyword']
result_df = merged_df[final_columns]

# 8. 결과 저장
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 9. 결과 미리보기
print(result_df.head())
print(f"\n총 {len(result_df)}개의 데이터가 처리되었습니다.")
print(f"결과 파일이 {output_path}에 저장되었습니다.")
