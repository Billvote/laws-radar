import sys
from pathlib import Path
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import re

# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# 데이터 불러오기
file_path = settings.BASE_DIR / 'geovote' / 'data' / 'bill_filtered_final.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi(model_type='knlm')

# 사용자 정의 고유 명사 추가
custom_nouns = [
    # 정부기관
    '대통령비서실', '국가안보실', '대통령경호처', '헌법상대통령자문기구', '국가안전보장회의',
    '민주평화통일자문회의', '국민경제자문회의', '국가과학기술자문회의', '감사원', '국가정보원',
    '방송통신위원회', '특별감찰관', '고위공직자범죄수사처', '국가인권위원회', '국무조정실',
    '국무총리비서실', '인사혁신처', '법제처', '식품의약품안전처', '공정거래위원회',
    '국민권익위원회', '금융위원회', '개인정보보호위원회', '원자력안전위원회', '기획재정부',
    '국세청', '관세청', '조달청', '통계청', '교육부', '과학기술정보통신부', '우주항공청',
    '외교부', '재외동포청', '통일부', '법무부', '검찰청', '국방부', '병무청', '방위사업청',
    '행정안전부', '경찰청', '소방청', '국가보훈부', '문화체육관광부', '국가유산청',
    '농림축산식품부', '농촌진흥청', '산림청', '산업통상자원부', '특허청', '보건복지부',
    '질병관리청', '환경부', '기상청', '고용노동부', '여성가족부', '국토교통부',
    '행정중심복합도시건설청', '새만금개발청', '해양수산부', '해양경찰청', '중소벤처기업부',
    # 국회/입법 용어
    '상임위원회', '법제사법위원회', '정무위원회', '기획재정위원회', '교육위원회',
    '과학기술정보방송통신위원회', '외교통일위원회', '국방위원회', '행정안전위원회',
    '문화체육관광위원회', '농림축산식품해양수산위원회', '산업통상자원중소벤처기업위원회',
    '보건복지위원회', '환경노동위원회', '국토교통위원회', '정보위원회', '여성가족위원회',
    '예산결산특별위원회', '특별위원회', '소위원회', '법안심사소위', '의안', '법률안',
    '예산안', '동의안', '승인안', '결의안', '건의안', '규칙안', '선출안', '발의', '제출',
    '제안', '제의', '의결', '부결', '폐기', '가결', '채택', '입법예고', '공포', '시행',
    '개정', '제정', '폐지', '일부개정', '전부개정',
    # 법률 분야
    '헌법', '민법', '형법', '상법', '행정법', '노동법', '세법', '환경법', '정보통신법',
    '금융법', '보건의료법', '교육법', '문화예술법', '농림법', '건설법', '해양법', '김영란법',
    '부정청탁금지법', '공직자윤리법', '정치자금법', '공직선거법', '전기통신사업법',
    '개인정보보호법', '국가유산수리기술위원회', '부가가치세법', '수입식품안전관리특별법',
    '다문화가족지원법',
    # 디지털/기술
    '인공지능', '빅데이터', '사물인터넷', '클라우드', '블록체인', '메타버스', '디지털플랫폼',
    '전자정부', '디지털전환', '사이버보안', '디지털뉴딜', '스마트시티', '디지털포용',
    '온라인플랫폼', '전자상거래',
    # 사회 현안
    '코로나19', '감염병', '백신', '방역', '사회적거리두기', '재난지원금', '기후변화',
    '미세먼지', '폐기물처리', '재활용', '순환경제', '젠더평등', '성희롱', '성폭력', '스토킹',
    '가정폭력', '디지털성범죄', '청년정책', '청년고용', '청년주택', '학자금대출', '교육격차'
]

for noun in custom_nouns:
    kiwi.add_user_word(noun, 'NNG', 9.0)

# 불용어 리스트 (목차 기호, 숫자형 목차, 법률 일반 용어 추가)
stopwords = {
    '조', '항', '호', '경우', '등', '수', '것', '이', '차', '후', '이상', '이하', '이내',
    '안', '소', '대', '점', '간', '곳', '해당', '차', '외', '경우', '나', '바', '시',
    '관련', '관하여', '대하여', '따라', '따른', '위하여', '의하여', '때', '각', '자', '인',
    '내', '중', '때문', '위해', '통해', '부터', '까지', '동안', '사이', '기준', '별도',
    '별첨', '별표', '제한', '특칙', '가능', '과정', '기반', '기존', '근거', '기능', '방식',
    '범위', '사항', '시점', '의한', '인한', '최근', '?', '년', '장', '해', '명', '날', '회',
    '동', '데', '국', '밖', '속', '식', '융', '밖', '규', '현행법',
    
    # 검색 결과 기반 추가 불용어 (약 200개)
    '가', '가령', '가지', '각각', '각자', '각종', '갖고말하자면', '같다', '같이', '거니와', 
    '거의', '것과 같이', '것들', '게다가', '겨우', '결과에 이르다', '결국', '결론을 낼 수 있다', 
    '겸사겸사', '고려하면', '고로', '곧', '과', '과연', '관계가 있다', '관계없이', '관련이 있다', 
    '관한', '관해서는', '구체적으로', '그', '그들', '그때', '그래', '그래도', '그래서', '그러나', 
    '그러니', '그러니까', '그러면', '그러므로', '그런 까닭에', '그런데', '그럼', '그럼에도 불구하고', 
    '그렇게 함으로써', '그렇지', '그렇지 않다면', '그렇지 않으면', '그렇지만', '그리고', '그리하여', 
    '그만이다', '그에 따르는', '그저', '근거로', '기타', '까지', '까지도', '나', '나머지는', '남들', 
    '너', '너희', '네', '논하지 않다', '다른', '다만', '다소', '다수', '다시 말하자면', '다음', 
    '다음에', '다음으로', '단지', '당신', '대하면', '대해 말하자면', '대해서', '더구나', '더군다나', 
    '더라도', '더불어', '동시에', '된바에야', '된이상', '두번째로', '둘', '등등', '따라', '따위', 
    '따지지 않다', '때가 되어', '또', '또한', '라 해도', '령', '로', '로 인하여', '로부터', '를', 
    '마음대로', '마저', '막론하고', '만 못하다', '만약', '만약에', '만은 아니다', '만일', '만큼', 
    '말하자면', '매', '모두', '무렵', '무슨', '물론', '및', '바꾸어말하면', '바꾸어서 말하면', 
    '바로', '밖에 안된다', '반대로', '반드시', '버금', '보는데서', '본대로', '부류의 사람들', 
    '불구하고', '사', '삼', '상대적으로 말하자면', '설령', '설마', '셋', '소생', '수', '습니까', 
    '습니다', '시간', '시작하여', '실로', '심지어', '아', '아니', '아니라면', '아니면', '아야', 
    '아울러', '안 그러면', '않기 위하여', '알 수 있다', '앞에서', '약간', '양자', '어', '어느', 
    '어디', '어때', '어떠한', '어떤', '어떻게', '어째서', '어쨋든', '언제', '얼마', '얼마만큼', 
    '엉엉', '에', '에게', '에서', '에이', '엔', '영', '예', '올', '와', '와르르', '와아', '왜', 
    '외에', '요', '우리', '원', '월', '위에서 서술한바와같이', '윙윙', '육', '으로', '으로서', 
    '으로써', '을', '응', '응당', '의', '의거하여', '의지하여', '의해', '이', '이르다', '이쪽', 
    '인젠', '일', '일것이다', '임에 틀림없다', '자기', '자기집', '자마자', '자신', '잠깐', '잠시', 
    '저', '저기', '저쪽', '전부', '전자', '정도에 이르다', '제', '제외하고', '조차', '좋아', '좍좍', 
    '주', '주저하지 않고', '줄', '중이다', '즈음', '즉', '지금', '지말고', '진짜로', '쪽으로', 
    '쯤', '차라리', '참', '첫번째로', '쳇', '콸콸', '쾅쾅', '쿵', '큼', '타다', '통하다', '틈타', 
    '팍', '퍽', '하', '하게될것이다', '하게하다', '하겠는가', '하고', '하고있었다', '하곤하였다', 
    '하구나', '하기 때문에', '하기만 하면', '하기보다는', '하기에', '하나', '하느니', '하는 것만', 
    '하는 편이 낫다', '하는것도', '하더라도', '하도다', '하도록시키다', '하도록하다', '하든지', 
    '하려고하다', '하마터면', '하면 할수록', '하면서', '하물며', '하여금', '하여야', '하지 않는다면', 
    '하지 않도록', '하지마', '하지마라', '하천', '하품', '한', '한 이유는', '한 후', '한다면', 
    '한다면 몰라도', '한데', '한마디', '한적이있다', '한켠으로는', '한편', '할 따름이다', '할 생각이다', 
    '할 줄 안다', '할 지경이다', '할 힘이다', '할때', '할만하다', '할망정', '할뿐', '함께', '해도', 
    '해봐요', '해서는', '해야한다', '해요', '했어요', '형식으로 쓰여', '혹시', '혹은', '혼자', '훨씬', 
    '휘익', '휴', '흐흐', '힘입어', 
    
    # 목차 기호
    '가.', '나.', '다.', '라.', '마.', '바.', '사.', '아.', '자.', '차.', '카.', '타.', '파.', '하.',
    '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.',
    '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩',
    'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅶ', 'Ⅷ', 'Ⅸ', 'Ⅹ',
    'i.', 'ii.', 'iii.', 'iv.', 'v.', 'vi.', 'vii.', 'viii.', 'ix.', 'x.',
    # 기타
    '.', '-', '·', '•', '※', '(', ')', '[', ']', '{', '}', '/', '\\',
    # 법률 일반 용어
    '법률', '법', '조례', '규정', '조항', '조문', '조치', '조정', '규칙', 
    '법안', '입법', '개정', '제정', '시행', '공포', '폐지', '일부개정', 
    '전부개정', '동의안', '승인안', '결의안', '건의안', '규칙안', '선출안', 
    '발의', '제출', '제안', '제의', '의결', '부결', '폐기', '가결', '채택', 
    '입법예고', '천만', '기관', '기간', 

    # 법률 특화 추가 불용어
    '법률', '법안', '입법', '개정', '제정', '시행', '공포', '폐지', '조례', '규정', 
    '조항', '조문', '조치', '조정', '규칙', '발의', '제출', '제안', '의결', '부결', 
    '가결', '채택', '심의', '처리', '안건', '의안', '상정', '의결', '재적', '위원', 
    '위원회', '소위원회', '본회의', '상임위', '특별위', '법제처', '입법예고',

}

def preprocess_text(text):
    """특수문자, 물음표 등 제거 및 공백 정규화와 복합어 처리"""
    text = str(text)
    text = text.replace('？', '').replace('?', '')
    # 한글, 영문, 숫자, 공백만 남기고 모두 제거
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', ' ', text)
    # 복합어 처리 (예: '수입 업' → '수입업')
    patterns = [
        (r'(수입)\s+(업)', r'\1업'),
        (r'(방제)\s+(업)', r'\1업'),
        (r'(제조)\s+(업)', r'\1업'),
        (r'(판매)\s+(업)', r'\1업'),
        (r'(서비스)\s+(업)', r'\1업'),
        (r'(유통)\s+(업)', r'\1업')
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    # 연속 공백을 단일 공백으로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_nouns(text):
    try:
        text = preprocess_text(text)
        tokens = kiwi.tokenize(text)
        words = []
        for token in tokens:
            # 명사(N)만 허용 (SN 태그 및 숫자 포함 단어 모두 제외)
            if not token.tag.startswith('N'):
                continue
            if any(c.isdigit() for c in token.form):
                continue
            if token.form in stopwords:
                continue
            words.append(token.form)
        return ' '.join(words)
    except Exception as e:
        print(f"Error processing text: {e}")
        return ''

# content 컬럼에서 직접 불용어 및 숫자 포함 단어 제거 (진행률 표시)
tqdm.pandas()
df['content'] = df['content'].progress_apply(extract_nouns)

# TF-IDF 벡터화 (더 구체적인 주제 추출을 위해 파라미터 조정)
vectorizer = TfidfVectorizer(
    max_df=0.75, 
    min_df=10,
    ngram_range=(1, 3),
    sublinear_tf=True
)
X = vectorizer.fit_transform(df['content'])

# K-means 클러스터링 (더 세분화된 주제 분류)
kmeans = KMeans(
    n_clusters=50,  # 더 세분화
    init='random',
    n_init=20,
    max_iter=500,
    random_state=42,
    algorithm='elkan'
)
df['cluster'] = kmeans.fit_predict(X)

# 클러스터별 대표 키워드 추출 및 라벨링
def get_top_keywords_per_cluster(X, labels, vectorizer, top_n=7):
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords = {}
    for cluster_num in np.unique(labels):
        idxs = np.where(labels == cluster_num)
        mean_tfidf = X[idxs].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        keywords = feature_names[top_idx]
        top_keywords[cluster_num] = ', '.join(keywords)
    return top_keywords

cluster_keywords = get_top_keywords_per_cluster(X, df['cluster'].values, vectorizer, top_n=7)
df['cluster_label'] = df['cluster'].apply(lambda x: f"{x} ({cluster_keywords[x]})")

# 결과 저장
output_path = settings.BASE_DIR / 'keword_clustering' / 'data' / 'clustered_bills_final3.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 클러스터별 대표 키워드 추출 함수
def get_top_keywords_per_cluster(X, labels, vectorizer, top_n=7):
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords = {}
    for cluster_num in np.unique(labels):
        idxs = np.where(labels == cluster_num)
        mean_tfidf = X[idxs].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        keywords = feature_names[top_idx]
        top_keywords[cluster_num] = ', '.join(keywords)
    return top_keywords

# 클러스터별 대표 키워드 출력
cluster_keywords = get_top_keywords_per_cluster(X, df['cluster'].values, vectorizer, top_n=7)
for cluster_num, keywords in cluster_keywords.items():
    print(f"클러스터 {cluster_num}: {keywords}")


print("✅ 클러스터링 완료 및 결과 저장")



# 클러스터링 수행
kmeans = KMeans(n_clusters=50, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 클러스터 분석
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_keywords = get_top_keywords_per_cluster(X, df['cluster'].values, vectorizer)

# 결과 출력
print("\n[최종 클러스터 현황]")
for cluster_num in sorted(cluster_keywords.keys()):
    print(f"클러스터 {cluster_num} ({cluster_counts[cluster_num]}개 법안)")
    print(f"  대표 키워드: {cluster_keywords[cluster_num]}\n")

# 결과 저장
df['cluster_size'] = df['cluster'].map(cluster_counts)
df.to_csv(output_path, index=False, encoding='utf-8-sig')