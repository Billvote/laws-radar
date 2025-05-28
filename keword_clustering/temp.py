# kiwi_analyze.py에 불용어/고유명사 추가 위해, 여기서 df 실행해서 샘플 확인

import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

import pandas as pd
import os 

# 모든 컬럼 내용을 줄이지 않고 전체 출력하도록 설정
pd.set_option('display.max_colwidth', None)

# 데이터 로드
file_path = settings.BASE_DIR / 'keword_clustering' / 'data' / 'clustered_bills.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')


df = df[['title', 'bill_number', 'cleaned', 'cluster']]

# 상위 
print(df['cleaned'].head(50))