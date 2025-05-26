import json
import pandas as pd
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# CSV 로드
df = pd.read_csv(settings.BASE_DIR / 'geovote' / 'data' / 'district_22.csv')
geo_path = settings.BASE_DIR / 'geo_visualization' / '2024_22_Elec.json'

# GeoJSON 로드
with open(geo_path, encoding="utf-8") as f:
    geojson = json.load(f)

# geometry 매핑해서 df에 붙이기
boundaries = {}
for feature in geojson["features"]:
    props = feature["properties"]
    geometry = feature["geometry"]
    sgg_code = str(props["SGG_Code"])
    boundaries[sgg_code] = json.dumps(geometry)

# df에 boundary 컬럼 추가
df["boundary"] = df["SGG_Code"].map(boundaries)

print(df.head())
# 새 CSV 저장
# df.to_csv("districts_with_boundary.csv", index=False)
