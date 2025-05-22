import pandas as pd
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 정당: 대수, 정당 이름
all_party_path = settings.MEMBER_INFO_DATA_DIR / 'all_member_18-22.csv'
df = pd.read_csv(all_party_path)
print(df.head())