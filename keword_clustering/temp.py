# kiwi_analyze.py에 불용어/고유명사 추가 위해, 여기서 df 실행해서 샘플 확인

import pandas as pd
import os 

# 모든 컬럼 내용을 줄이지 않고 전체 출력하도록 설정
pd.set_option('display.max_colwidth', None)

base_dir = 'C:/Users/1-08/OneDrive/Desktop/DAMF2/Final_PJT/'
file_path = os.path.join(base_dir, 'keword_clustering/data/clustered_bills_22.csv')
df = pd.read_csv(file_path)

df = df[['BILL_NAME', 'BILL_NO', 'cleaned', 'cluster']]

print(df['cleaned'].head(50))