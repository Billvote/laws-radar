import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import os

# 데이터 불러오기
# 기본 경로 설정

file_path = settings.BASE_DIR / 'geovote' / 'data' / 'bill_filtered_final.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')


# 형태소 분석기 초기화
kiwi = Kiwi()

# ✅ 사용자 정의 고유 명사 추가
custom_nouns = [
    # 정부 부처
    '대통령비서실', '국가안보실', '대통령경호처', '헌법상대통령자문기구', '국가안전보장회의', '민주평화통일자문회의', '국민경제자문회의',
    '국가과학기술자문회의', '국민경제자문회의', '국가과학기술자문회의', '감사원', '국가정보원', '방송통신위원회', '특별감찰관',
    '고위공직자범죄수사처', '국가인권위원회', '국무조정실', '국무총리비서실', '인사혁신처','법제처', '식품의약품안전처',
    '공정거래위원회', '국민권익위원회', '금융위원회', '개인정보보호위원회', '원자력안전위원회', '기획재정부', '국세청', '관세청',
    '조달청', '통계청', '교육부', '과학기술정보통신부', '우주항공청', '외교부', '재외동포청', '통일부', '법무부', '검찰청', '검찰청',
    '국방부', '병무청', '방위사업청', '행정안전부', '경찰청', '소방청', '국가보훈부', '문화체육관광부', '국가유산청', '농림축산식품부',
    '농촌진흥청', '산림청', '산업통상자원부', '특허청', '보건복지부', '질병관리청', '환경부', '기상청', '고용노동부', '여성가족부', '국토교통부',
    '행정중심복합도시건설청', '새만금개발청', '해양수산부', '해양경찰청','중소벤처기업부',

    # 의안별 일반명사
    '전기통신사업법', '개인정보보호법', '국가유산수리기술위원회', '국가유산수리기술자', '부가가치세법', '국가유산수리업자', '국가유산수리', '국가유산수리기능자',
    '게임물', '제공자', '지방자치단체', '산업통상자원중소벤처위원회', '수입식품안전관리특별법', '북한이탈주민', '행정처분', '다문화가족지원법', '공용윤리위원회',
    '인공지능',

]

for noun in custom_nouns:
    kiwi.add_user_word(noun, 'NNG')  # 또는 'NNP' (고유 명사)도 가능

# 사용자 정의 불용어 리스트 (원하는 단어 추가 가능)
stopwords = {
    '조', '항', '호', '경우', '등', '수', '것', '이', '차', '후', '이상', '이하', '이내'
    '안', '소', '대', '점', '간', '곳', '해당', '차', '외', '경우', '나', '바', '시'
    }


# 명사와 숫자를 추출하는 함수
def extract_nouns(text):
    try:
        tokens = kiwi.tokenize(str(text))
        
        # 명사(N으로 시작) 또는 숫자(SN)만 추출
        words = [
            token.form
            for token in tokens
            if token.tag.startswith('N') or token.tag == 'SN'
        ]
        
        # 불용어 제거
        filtered_words = [word for word in words if word not in stopwords]
        
        return ' '.join(filtered_words)
    except:
        return ''

# 전처리: 명사 추출 후 새로운 컬럼에 저장
df['cleaned'] = df['content'].apply(extract_nouns)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df['cleaned'])

# KMeans 클러스터링 (클러스터 수는 원하는 만큼 설정)
k = 5  # 예시로 5개 주제
model = KMeans(n_clusters=k, random_state=42)
df['cluster'] = model.fit_predict(X)

# 결과 확인
print(df[['title', 'cluster']].head())

# 저장
output_path = settings.BASE_DIR / 'keword_clustering' / 'data' / 'clustered_bills.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("✅ 클러스터링 결과 저장 완료")