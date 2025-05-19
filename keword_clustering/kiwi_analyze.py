from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import os

# 데이터 불러오기
# 기본 경로 설정
base_dir = 'C:/Users/1-08/OneDrive/Desktop/DAMF2/Final_PJT/'
file_path = os.path.join(base_dir, 'merged/data/summary+vote_22.csv')
df = pd.read_csv(file_path)

# 형태소 분석기 초기화
kiwi = Kiwi()

# 명사만 추출하는 함수
def extract_nouns(text):
    try:
        tokens = kiwi.tokenize(str(text))
        nouns = [token.form for token in tokens if token.tag.startswith('N')]  # 명사만
        return ' '.join(nouns)
    except:
        return ''

# 전처리: 명사 추출 후 새로운 컬럼에 저장
df['cleaned'] = df['summary'].apply(extract_nouns)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df['cleaned'])

# KMeans 클러스터링 (클러스터 수는 원하는 만큼 설정)
k = 5  # 예시로 5개 주제
model = KMeans(n_clusters=k, random_state=42)
df['cluster'] = model.fit_predict(X)

# 결과 확인
print(df[['BILL_NAME', 'cluster']].head())

# 저장 (선택)
df.to_csv('clustered_bills.csv', index=False, encoding='utf-8-sig')
print("✅ 클러스터링 결과 저장 완료: clustered_bills.csv")