# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import google.generativeai as genai
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from gensim.corpora import Dictionary

tqdm.pandas()

# 1. 사용자 정의 필터 설정 --------------------------------------------------------
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

initial_stopwords = frozenset({
    '조', '항', '호', '경우', '등', '수', '것', '이', '차', '후', '이상', '이하', '이내',
    '안', '소', '대', '점', '간', '곳', '해당', '외', '나', '바', '시', '관련', '관하여',
    '대하여', '따라', '따른', '위하여', '의하여', '때', '각', '자', '인', '내', '중',
    '때문', '위해', '통해', '부터', '까지', '동안', '사이', '기준', '별도', '별첨', '별표',
    '제한', '특칙', '가능', '과정', '기반', '기존', '근거', '기능', '방식', '범위', '사항',
    '시점', '최근', '년', '장', '해', '명', '날', '회', '동', '데', '국', '밖', '속', '식',
    '규', '현행법', '직', '범', '만', '입', '신',
})

initial_excluded_terms = frozenset({
    '주요', '수사', '관련', '사항', '정책', '대상', '방안', '추진', '강화', '개선', '지원',
    '확대', '조치', '필요', '현황', '기반', '과정', '기존', '근거', '기능', '방식', '범위',
    '활동', '운영', '관리', '실시', '확보', '구성', '설치', '지정', '계획', '수립',
})

preserve_terms = frozenset({'법률', '법안', '입법', '개정', '제정', '시행', '공포', '폐지', '조례', '규정', '조항', '의결'})
excluded_bigrams = frozenset({'교육 실시', '징역 벌금', '수립 시행', '운영 관리'})

# 2. 법령 구조 패턴 제거 함수 -----------------------------------------------------
def remove_law_structure_phrases(text):
    patterns = [
        r'제\d+조의?\d*(?:제\d+항)?(?:제\d+호)?',
        r'안\s*제\d+조의?\d*(?:제\d+항)?(?:제\d+호)?',
        r'\d+년\s*\d+월\s*\d+일',
        r'\d+만\s*\d+천?\s*\d+명',
        r'\d+%',
        r'\b(?:누구나|지니고|유사한|기준|약)\b',
        r'신설', r'정비', r'조정', r'인용조문', r'정비\s*\(.*?\)', r'안'
    ]
    combined = '|'.join(patterns)
    text = re.sub(combined, ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# 3. Gemini 임베딩 생성 함수 -----------------------------------------------------
def get_gemini_embeddings(texts, model_name="gemini-embedding-exp-03-07", task_type="CLUSTERING", max_workers=1):
    embeddings = []
    def embed_one(text):
        try:
            response = genai.embed_content(
                model=model_name,
                content=text,
                task_type=task_type
            )
            return response['embedding']
        except Exception as e:
            print(f"⚠️ 임베딩 오류: {str(e)[:100]}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_one, text) for text in texts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Gemini 임베딩 생성"):
            if (emb := future.result()) is not None:
                embeddings.append(emb)
    
    print(f"✅ 생성된 임베딩 수: {len(embeddings)}")
    return embeddings

# 4. Gemini 전처리 함수 ---------------------------------------------------------
def gemini_tokenize_and_filter(text, model):
    prompt = f"""
[한국어 지시사항]
아래 법안 텍스트에서
1. 사용자 정의 명사 {custom_nouns}는 반드시 보존
2. 초기 불용어 {list(initial_stopwords)}와 제외 단어 {list(initial_excluded_terms)}는 제거
3. 제외 바이그램 {list(excluded_bigrams)}는 전체 삭제
4. 보존 용어 {list(preserve_terms)}는 무조건 유지
5. 결과를 공백 구분 문자열로 반환

텍스트:
{text[:2000]}
"""
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            tokens = response.text.strip()
            if tokens and len(tokens) > 1:
                return tokens
        except Exception:
            time.sleep(1)
    return ""

def parallel_gemini_tokenize_and_filter(texts, model, max_workers=1):
    results = [None] * len(texts)
    def process_one(i, t):
        return (i, gemini_tokenize_and_filter(t, model))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one, i, t) for i, t in enumerate(texts)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Gemini 병렬 전처리"):
            i, tokens = f.result()
            results[i] = tokens
    
    return results

# 5. LDA 기반 최적 군집수 탐색 ---------------------------------------------------
def find_optimal_n_topics_lda_fast(X, texts, dictionary, corpus, vectorizer):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics import silhouette_score

    # 1단계: 대략적 범위 탐색
    n_range_coarse = range(10, 201, 50)
    best_score_coarse = -np.inf
    best_n_coarse = 10

    for n_topics in tqdm(n_range_coarse, desc="1단계: 대략적 군집수 탐색"):
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            batch_size=256,
            max_iter=15,
            random_state=42,
            n_jobs=1
        )
        topic_dist = lda.fit_transform(X)
        silhouette = silhouette_score(X, topic_dist.argmax(axis=1))
        if silhouette > best_score_coarse:
            best_score_coarse = silhouette
            best_n_coarse = n_topics

    # 2단계: 상세 범위 탐색
    start = max(10, best_n_coarse - 40)
    end = min(200, best_n_coarse + 40)
    n_range_fine = range(start, end+1, 10)

    def train_lda(n):
        lda = LatentDirichletAllocation(
            n_components=n,
            learning_method='online',
            batch_size=256,
            max_iter=15,
            random_state=42,
            n_jobs=1
        )
        lda.fit(X)
        return lda

    print("2단계: 상세 군집수 탐색 (병렬)")
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(train_lda, n): n for n in n_range_fine}
        models = {}
        for future in tqdm(as_completed(futures), total=len(futures)):
            n = futures[future]
            models[n] = future.result()

    # 최적 모델 선정
    best_score = -np.inf
    best_n = 10
    for n, model in models.items():
        topic_dist = model.transform(X)
        silhouette = silhouette_score(X, topic_dist.argmax(axis=1))
        if silhouette > best_score:
            best_score = silhouette
            best_n = n

    return best_n

# 6. Gemini 기반 군집수 결정 -----------------------------------------------------
def get_optimal_clusters_with_gemini(embeddings, sample_texts, model):
    prompt = f'''[한국어 지시사항]
샘플 텍스트: {sample_texts[:2000]}
최적 클러스터 수 1개 추천 (10~200):
{{"optimal_clusters": 정수}}'''
    
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text.strip())
            if 10 <= (n := result["optimal_clusters"]) <= 200:
                return n
        except Exception:
            time.sleep(1)
    return None

# 7. 클러스터 키워드 추출 --------------------------------------------------------
def gemini_cluster_keywords(cluster_id, texts, model):
    prompt = f'''[한국어 지시사항]
법안 샘플: {texts[:2000]}
4개 키워드 추출:
- 법률 조문/정책명 우선
- JSON 배열로만 반환'''
    
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            if (match := re.search(r'\[".*?"(?:,\s*".*?")*\]', response.text)):
                return json.loads(match.group(0))[:4]
        except Exception as e:
            print(f"⚠️ 키워드 추출 오류 (클러스터 {cluster_id}): {str(e)}")
            time.sleep(1)
    
    # Fallback
    vectorizer = TfidfVectorizer(max_features=20)
    X = vectorizer.fit_transform([texts])
    return vectorizer.get_feature_names_out()[:4].tolist()

# 8. 메인 실행 로직 --------------------------------------------------------------
if __name__ == '__main__':
    # 초기화
    GEMINI_API_KEY = "AIzaSyA8M00iSzCK1Lvc5YfxamYgQf-Lh4xh5R0"
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # 데이터 로드
    file_path = Path(r"C:/Users/1-02/Desktop/DAMF2/laws-radar/geovote/data/bill_filtered_final.csv")
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 1. 전처리
    print("🔄 법령 구조 패턴 제거 중...")
    df['content'] = df['content'].apply(remove_law_structure_phrases)

    print("🔄 Gemini 전처리 중...")
    df['content'] = parallel_gemini_tokenize_and_filter(df['content'].tolist(), model, 1)

    # 2. 임베딩 (유효한 데이터 필터링)
    print("🔄 임베딩 생성 중...")
    embeddings = get_gemini_embeddings(df['content'].tolist())
    
    # 유효한 인덱스 추출
    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    filtered_df = df.iloc[valid_indices].copy()
    valid_embeddings = np.array([emb for emb in embeddings if emb is not None])

    # ▼▼▼ 수정된 부분: 배열 크기 확인 ▼▼▼
    if valid_embeddings.size == 0:
        print("❌ 임베딩 실패: 생성된 임베딩이 없습니다")
        sys.exit(1)
    # ▲▲▲

    # 3. 군집수 결정
    print("🔎 군집수 분석 중...")
    n_clusters = get_optimal_clusters_with_gemini(
        valid_embeddings, 
        ' '.join(filtered_df.sample(min(100, len(filtered_df)))['content']), 
        model
    )
    
    if not n_clusters:
        print("⚠️ LDA 방식으로 전환")
        vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=(1,3), max_features=5000)
        X = vectorizer.fit_transform(filtered_df['content'])
        texts = [doc.split() for doc in filtered_df['content']]
        n_clusters = find_optimal_n_topics_lda_fast(X, texts, Dictionary(texts), [], vectorizer)
    
    n_samples = len(valid_embeddings)
    n_clusters = min(n_clusters, n_samples)
    print(f"✅ 최종 군집수: {n_clusters} (샘플 수: {n_samples})")

    # 4. 클러스터링
    print("🔄 클러스터링 중...")
    filtered_df['topic'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(valid_embeddings)

    # 5. 원본 데이터프레임에 병합
    df = df.merge(filtered_df[['topic']], how='left', left_index=True, right_index=True)
    df.rename(columns={'topic_y': 'topic'}, inplace=True)
    df['topic'] = df['topic'].fillna(-1).astype(int)  # 임베딩 실패 행은 -1로 표시

    # 6. 키워드 추출
    print("🔄 키워드 추출 중...")
    cluster_texts = filtered_df.groupby('topic')['content'].apply(lambda x: ' '.join(x.sample(min(10, len(x)))))
    topic_labels = {}
    
    with ThreadPoolExecutor(1) as executor:
        futures = {executor.submit(gemini_cluster_keywords, cid, txt, model): cid 
                  for cid, txt in cluster_texts.items()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            cid = futures[future]
            topic_labels[cid] = future.result()
            print(f"✅ 클러스터 {cid} 키워드: {topic_labels[cid]}")

    # 7. 결과 병합
    df['topic_label'] = df['topic'].map(lambda x: ', '.join(topic_labels.get(x, ['기타'])))

    # 8. 저장
    output_path = Path('data/bill_gemini_clustering.csv')
    df[['age', 'title', 'bill_id', 'bill_number', 'content', 'topic', 'topic_label']].to_csv(
        output_path, index=False, encoding='utf-8-sig'
    )
    print(f"✅ 저장 완료: {output_path}")
