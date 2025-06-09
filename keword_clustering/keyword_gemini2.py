# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
from kiwipiepy import Kiwi
from tqdm import tqdm
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import google.generativeai as genai
import time
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

tqdm.pandas()

# 시스템 경로 설정
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

# Gemini API 초기화
GEMINI_API_KEY = "AIzaSyA8M00iSzCK1Lvc5YfxamYgQf-Lh4xh5R0"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# 데이터 로드
file_path = settings.BASE_DIR / 'geovote' / 'data' / 'bill_filtered_final.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi(model_type='knlm')

# 사용자 정의 명사 목록
custom_nouns = [
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
    '상임위원회', '법제사법위원회', '정무위원회', '기획재정위원회', '교육위원회',
    '과학기술정보방송통신위원회', '외교통일위원회', '국방위원회', '행정안전위원회',
    '문화체육관광위원회', '농림축산식품해양수산위원회', '산업통상자원중소벤처기업위원회',
    '보건복지위원회', '환경노동위원회', '국토교통위원회', '정보위원회', '여성가족위원회',
    '예산결산특별위원회', '특별위원회', '소위원회', '법안심사소위', '의안', '법률안',
    '예산안', '동의안', '승인안', '결의안', '건의안', '규칙안', '선출안', '발의', '제출',
    '제안', '제의', '의결', '부결', '폐기', '가결', '채택', '입법예고', '공포', '시행',
    '개정', '제정', '폐지', '일부개정', '전부개정',
    '헌법', '민법', '형법', '상법', '행정법', '노동법', '세법', '환경법', '정보통신법',
    '금융법', '보건의료법', '교육법', '문화예술법', '농림법', '건설법', '해양법', '김영란법',
    '부정청탁금지법', '공직자윤리법', '정치자금법', '공직선거법', '전기통신사업법',
    '개인정보보호법', '국가유산수리기술위원회', '부가가치세법', '수입식품안전관리특별법',
    '다문화가족지원법',
    '인공지능', '빅데이터', '사물인터넷', '클라우드', '블록체인', '메타버스', '디지털플랫폼',
    '전자정부', '디지털전환', '사이버보안', '디지털뉴딜', '스마트시티', '디지털포용',
    '온라인플랫폼', '전자상거래',
    '코로나19', '감염병', '백신', '방역', '사회적거리두기', '재난지원금', '기후변화',
    '미세먼지', '폐기물처리', '재활용', '순환경제', '젠더평등', '성희롱', '성폭력', '스토킹',
    '가정폭력', '디지털성범죄', '청년정책', '청년고용', '청년주택', '학자금대출', '교육격차',
]
for noun in custom_nouns:
    kiwi.add_user_word(noun, 'NNG', 9.0)

# 불용어 및 제외 단어 정의
stopwords = {
    '조', '항', '호', '경우', '등', '수', '것', '이', '차', '후', '이상', '이하', '이내',
    '안', '소', '대', '점', '간', '곳', '해당', '차', '외', '경우', '나', '바', '시',
    '관련', '관하여', '대하여', '따라', '따른', '위하여', '의하여', '때', '각', '자', '인',
    '내', '중', '때문', '위해', '통해', '부터', '까지', '동안', '사이', '기준', '별도',
    '별첨', '별표', '제한', '특칙', '가능', '과정', '기반', '기존', '근거', '기능', '방식',
    '범위', '사항', '시점', '최근', '?', '년', '장', '해', '명', '날', '회',
    '동', '데', '국', '밖', '속', '식', '융', '밖', '규', '현행법', '직', '범', '만', '입', '직',
    '신',
    '가', '가령', '가지', '각각', '각자', '각종', '갖고말하자면', '같다', '같이', '거니와',
    '거의', '것과 같이', '것들', '게다가', '겨우', '결과에 이르다', '결국', '결론을 낼 수 있다',
    '겸사겸사', '고려하면', '고로', '곧', '과', '과연', '관계가 있다', '관계없이', '관련이 있다',
    '관한', '관해서는', '구체적으로', '그', '그들', '그때', '그래', '그래도', '그래서', '그러나',
    '그러니', '그러니까', '그러면', '그러므로', '그런 까닭에', '그런데', '그럼', '그럼에도 불구하고',
    '그렇게 함으로써', '그렇지', '그렇지 않다면', '그렇지 않으면', '그렇지만', '그리고', '그리하여',
    '그만이다', '그에 따르는', '그저', '근거로', '기타', '까지', '까지도', '나', '나머지는', '남들',
    '너', '너희', '네', '논하지 않다', '다른', '다만', '다소', '다수', '다시 말하자면', '다음',
    '다음에', '다음으로', '단지', '당신', '대하면', '대해 말하자면', '대해서', '더구나', '더군다나',
    '더라도', '더불어', '동시에', '된바에야', '된이상', '두번째로', '둘', '등등', '따라', '� yawn',
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
    '할 줄 안다', '할 지경이다', '할때', '할만하다', '할망정', '할뿐', '함께', '해도',
    '해봐요', '해서는', '해야한다', '해요', '했어요', '형식으로 쓰여', '혹시', '혹은', '혼자', '훨씬',
    '휘익', '휴', '흐흐', '힘입어',
    '가.', '나.', '다.', '라.', '마.', '바.', '사.', '아.', '자.', '차.', '카.', '타.', '파.', '하.',
    '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.',
    '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩',
    'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅶ', 'Ⅷ', 'Ⅸ', 'Ⅹ',
    'i.', 'ii.', 'iii.', 'iv.', 'v.', 'vi.', 'vii.', 'viii.', 'ix.', 'x.',
    '법률', '법', '조례', '규정', '조항', '조문', '조치', '조정', '규칙',
    '법안', '입법', '개정', '제정', '시행', '공포', '폐지', '일부개정', '전부개정', '동의안', '승인안', '결의안', '건의안', '규칙안', '선출안',
    '발의', '제출', '제안', '제의', '의결', '부결', '폐기', '가결', '채택',
    '입법예고', '천만', '기관', '기간'
}
preserve_terms = {'법률', '법안', '입법', '개정', '제정', '시행', '공포', '폐지', '조례', '규정', '조항', '의결'}
stopwords = {term for term in stopwords if term not in preserve_terms}
excluded_terms = {
    '주요', '수사', '관련', '사항', '정책', '대상', '방안', '추진', '강화', '개선', '지원',
    '확대', '조치', '필요', '현황', '기반', '과정', '기존', '근거', '기능', '방식', '범위'
}
excluded_bigrams = {'교육 실시', '징역 벌금', '수립 시행'}

# 전처리 함수 (최적화)
def preprocess_text(text):
    text = str(text).replace('？', '').replace('?', '')
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', ' ', text)
    patterns = [(r'(\w+)\s+(업)', r'\1업')]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_nouns(texts):
    """벡터화된 형태소 분석"""
    results = []
    for text in texts:
        try:
            tokens = kiwi.tokenize(preprocess_text(text))
            words = [
                token.form for token in tokens
                if token.tag.startswith('N') and
                not any(c.isdigit() for c in token.form) and
                token.form not in excluded_terms and
                token.form not in stopwords
            ]
            results.append(' '.join(words))
        except Exception as e:
            print(f"전처리 오류: {e}")
            results.append('')
    return results

def remove_single_char_words(texts):
    """단일 문자 제거 (벡터화)"""
    return [' '.join(word for word in text.split() if len(word) > 1) for text in texts]

# 데이터 전처리
df['content'] = extract_nouns(df['content'].tolist())
df['content'] = remove_single_char_words(df['content'].tolist())

# LDA용 벡터화
vectorizer = CountVectorizer(
    max_df=0.5,
    min_df=10,
    ngram_range=(1, 2),
    max_features=3000,
    token_pattern=r"(?u)\b\w+\b"
)
X = vectorizer.fit_transform(df['content'])

# 실루엣 점수로 최적 클러스터 수 결정
def find_optimal_n_topics(X, min_topics=10, max_topics=50, step=5):
    best_n_topics = min_topics
    best_score = -1
    print("\n🔍 실루엣 점수로 최적 클러스터 수 계산 중...")
    for n_topics in tqdm(range(min_topics, max_topics + 1, step)):
        try:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='batch',
                random_state=42,
                n_jobs=1  # 단일 스레드 사용
            )
            topic_dist = lda.fit_transform(X)
            score = silhouette_score(X, topic_dist.argmax(axis=1))
            print(f"n_topics={n_topics}, 실루엣 점수: {score:.4f}")
            if score > best_score:
                best_score = score
                best_n_topics = n_topics
        except Exception as e:
            print(f"n_topics={n_topics} 계산 오류: {e}")
            continue
    print(f"✅ 최적 클러스터 수: {best_n_topics} (실루엣 점수: {best_score:.4f})")
    return best_n_topics

# 최적 클러스터 수로 LDA 훈련
n_topics = find_optimal_n_topics(X, min_topics=10, max_topics=50, step=5)
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10,
    learning_method='batch',
    random_state=42,
    n_jobs=1  # 단일 스레드 사용
)
lda.fit(X)
df['topic'] = lda.transform(X).argmax(axis=1)

# 토픽별 키워드 추출 함수
def get_top_words(model, feature_names, n_top_words=12):
    topic_keywords = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        keywords = []
        for i in top_indices:
            word = feature_names[i]
            if (word in stopwords or word in excluded_terms or len(word) == 1 or
                word in excluded_bigrams or
                any(part in stopwords or part in excluded_terms for part in word.split())):
                continue
            keywords.append(word)
            if len(keywords) >= 4:
                break
        topic_keywords[topic_idx] = keywords
    return topic_keywords

# Gemini 키워드 정제 함수
@lru_cache(maxsize=1000)
def refine_keywords_with_gemini_cached(keywords_tuple):
    keywords = list(keywords_tuple)
    prompt = f'''[한국어 지시사항]
다음 키워드 목록에서 법령 관련성을 고려해 가장 중요한 4개의 핵심 키워드만 콤마로 구분하여 출력:
- 키워드는 주어진 법령 문서와 주제적으로 연관성이 높아야 함
- 중복/유사어 통합
- 일반적 단어 제외
- 법령 용어 우선선택
- 결과는 정확히 4개의 키워드로, 콤마로 구분된 문자열로 반환
- 불용어 및 제외 단어, 바이그램은 포함시키지 않음
- 추가 설명이나 메타 정보는 절대 포함시키지 않음

원본 키워드: {", ".join(keywords)}
'''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response.text:
                cleaned = response.text.strip().replace('```', '').replace('\n', '')
                if cleaned.startswith('{') or cleaned.startswith('['):
                    try:
                        kw_list = json.loads(cleaned)
                        if isinstance(kw_list, dict):
                            kw_list = kw_list.get('refined_keywords', [])
                        kw_list = [str(kw).strip() for kw in kw_list]
                    except json.JSONDecodeError:
                        kw_list = [kw.strip() for kw in cleaned.strip('[]{}').split(',')]
                else:
                    kw_list = [kw.strip() for kw in cleaned.split(',') if kw.strip()]

                filtered_keywords = []
                for kw in kw_list:
                    if (kw in stopwords or kw in excluded_terms or kw in excluded_bigrams or
                        len(kw) <= 1 or
                        any(part in stopwords or part in excluded_terms for part in kw.split()) or
                        any(kw in existing for existing in filtered_keywords)):
                        continue
                    filtered_keywords.append(kw)
                    if len(filtered_keywords) >= 4:
                        break

                if len(filtered_keywords) >= 4:
                    return ", ".join(filtered_keywords[:4])
                for kw in keywords:
                    if (kw not in filtered_keywords and
                        kw not in stopwords and
                        kw not in excluded_terms and
                        kw not in excluded_bigrams and
                        len(kw) > 1 and
                        not any(part in stopwords or part in excluded_terms for part in kw.split()) and
                        not any(kw in existing for existing in filtered_keywords)):
                        filtered_keywords.append(kw)
                    if len(filtered_keywords) >= 4:
                        break
                if len(filtered_keywords) >= 4:
                    return ", ".join(filtered_keywords[:4])
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ 키워드 정제 오류 (시도 {attempt+1}/{max_retries}): {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue

    # Fallback: 원본 키워드에서 4개 선택
    filtered_keywords = [
        kw for kw in keywords
        if kw not in stopwords and
        kw not in excluded_terms and
        kw not in excluded_bigrams and
        len(kw) > 1 and
        not any(part in stopwords or part in excluded_terms for part in kw.split())
    ]
    if len(filtered_keywords) >= 4:
        return ", ".join(filtered_keywords[:4])
    extra_keywords = [
        feature_names[i] for i in lda.components_[0].argsort()[::-1][12:24]
        if feature_names[i] not in stopwords and
        feature_names[i] not in excluded_terms and
        feature_names[i] not in excluded_bigrams and
        len(feature_names[i]) > 1 and
        not any(part in stopwords or part in excluded_terms for part in feature_names[i].split())
    ]
    filtered_keywords.extend(extra_keywords)
    return ", ".join(filtered_keywords[:4])

def refine_keywords_parallel(topic_keywords):
    topic_labels_refined = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_topic = {
            executor.submit(
                refine_keywords_with_gemini_cached,
                tuple(keywords)
            ): topic_id
            for topic_id, keywords in topic_keywords.items()
        }
        for future in tqdm(as_completed(future_to_topic), total=len(future_to_topic)):
            topic_id = future_to_topic[future]
            try:
                refined = future.result()
                topic_labels_refined[topic_id] = refined
            except Exception as e:
                print(f"⚠️ 토픽 {topic_id} 처리 오류: {str(e)[:100]}")
                keywords = topic_keywords[topic_id]
                filtered_keywords = [
                    kw for kw in keywords
                    if kw not in stopwords and
                    kw not in excluded_terms and
                    kw not in excluded_bigrams and
                    len(kw) > 1 and
                    not any(part in stopwords or part in excluded_terms for part in kw.split())
                ]
                if len(filtered_keywords) < 4:
                    extra_keywords = [
                        feature_names[i] for i in lda.components_[topic_id].argsort()[::-1][12:24]
                        if feature_names[i] not in stopwords and
                        feature_names[i] not in excluded_terms and
                        feature_names[i] not in excluded_bigrams and
                        len(feature_names[i]) > 1 and
                        not any(part in stopwords or part in excluded_terms for part in feature_names[i].split())
                    ]
                    filtered_keywords.extend(extra_keywords[:4-len(filtered_keywords)])
                topic_labels_refined[topic_id] = ", ".join(filtered_keywords[:4])
    return topic_labels_refined

# 토픽별 키워드 추출 및 정제
feature_names = np.array(vectorizer.get_feature_names_out())
topic_keywords = get_top_words(lda, feature_names, n_top_words=12)

print("\n🔧 Gemini API를 이용한 키워드 정제 시작 (병렬 처리)...")
topic_labels_refined = refine_keywords_parallel(topic_keywords)

# 모든 토픽에 키워드 할당
for topic_id in range(n_topics):
    if topic_id not in topic_labels_refined:
        topic = lda.components_[topic_id]
        keywords = [
            feature_names[i] for i in topic.argsort()[::-1][:12]
            if feature_names[i] not in stopwords and
            feature_names[i] not in excluded_terms and
            feature_names[i] not in excluded_bigrams and
            len(feature_names[i]) > 1 and
            not any(part in stopwords or part in excluded_terms for part in feature_names[i].split())
        ]
        topic_labels_refined[topic_id] = ", ".join(keywords[:4])

# 결과 매핑
df['topic_label'] = df['topic'].map(topic_labels_refined)
df['topic_label'] = df['topic_label'].fillna(", ".join(feature_names[lda.components_[0].argsort()[::-1][:4]]))

# 최종 저장
output_path = settings.BASE_DIR / 'keword_clustering' / 'data' / 'bill_keyword_clustering_refined.csv'
df[['age', 'title', 'bill_id', 'bill_number', 'content', 'topic', 'topic_label']].to_csv(
    output_path,
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
    encoding='utf-8-sig'
)

print("✅ 처리 완료! 파일: bill_keyword_clustering_refined.csv")