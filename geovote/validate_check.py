import pandas as pd
import json
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings


file_path = settings.BASE_DIR / 'geovote' / 'data' / 'boundary.csv'
df = pd.read_csv(file_path)
errors = []

for i, raw in enumerate(df['boundary']):
    try:
        json.loads(raw)
    except Exception as e:
        errors.append((i, e))

print(f"총 오류 행: {len(errors)}")
for idx, err in errors:
    print(f"행 {idx}: {err}")
