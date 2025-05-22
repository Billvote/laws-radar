import pandas as pd
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 정당: 대수, 정당 이름
# 1. 필요한 CSV 불러오기
all_party_path = settings.MEMBER_INFO_DATA_DIR / 'all_member_18-22.csv'
df = pd.read_csv(all_party_path)

# 2. 필요한 컬럼 선택 및 이름 정리
df = df[['대수', '정당']].rename(columns={'대수': 'age', '정당': 'party'})

# 3. 20~22대 필터링
filtered = df[df['age'].isin([20, 21, 22])]

# # 4. 중복 제거 (정당명 기준)
# unique_parties = filtered['party'].dropna().drop_duplicates().reset_index(drop=True)

# # 5. DataFrame으로 변환
# unique_df = pd.DataFrame({'party': unique_parties})

# # 6. 저장
# output_path = settings.BASE_DIR / 'member_info' / 'data' / 'unique_parties_20_22.csv'
# unique_df.to_csv(output_path, index=False)

# print(f"✅ 유니크 정당 CSV 저장 완료: {output_path}")

# print(filtered)
# print(filtered.groupby('age').value_counts())
# print(filtered.groupby('age')['party'].count())
# print(filtered.groupby('age')['party'].unique())