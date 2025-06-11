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
from concurrent.futures import ThreadPoolExecutor, as_completed
from gensim.corpora import Dictionary

tqdm.pandas()

# 1. Gemini 임베딩 생성 함수
def get_gemini_embeddings(texts, model_name="models/embedding-001", task_type="CLUSTERING", max_workers=4):
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
            emb = future.result()
            if emb is not None:
                embeddings.append(emb)
    print(f"✅ 생성된 임베딩 수: {len(embeddings)}")
    return embeddings

# 2. 텍스트 전처리 함수
def gemini_tokenize_and_filter(text, model):
    prompt = f"""
[한국어 지시사항]
아래 법안 텍스트에서
1. 일반 불용어/중복단어 제거
2. 법률 용어 중심 토큰화
3. 결과를 공백 구분 문자열로 반환
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

def parallel_gemini_tokenize_and_filter(texts, model, max_workers=4):
    results = [None] * len(texts)
    def process_one(i, t):
        return (i, gemini_tokenize_and_filter(t, model))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one, i, t) for i, t in enumerate(texts)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Gemini 병렬 전처리"):
            i, tokens = f.result()
            results[i] = tokens
    return results

# 3. LDA 기반 최적 군집수 탐색 (속도 최적화)
def find_optimal_n_topics_lda_fast(X, texts, dictionary, corpus, vectorizer):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics import silhouette_score

    # 1단계: 대략적 범위 탐색 (50단위)
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

    # 2단계: 상세 범위 탐색 (10단위, 병렬)
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
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(train_lda, n): n for n in n_range_fine}
        models = {}
        for future in tqdm(as_completed(futures), total=len(futures)):
            n = futures[future]
            models[n] = future.result()

    # 최적 모델 선정 (실루엣 점수만)
    best_score = -np.inf
    best_n = 10
    for n, model in models.items():
        topic_dist = model.transform(X)
        silhouette = silhouette_score(X, topic_dist.argmax(axis=1))
        if silhouette > best_score:
            best_score = silhouette
            best_n = n

    return best_n

# 4. Gemini 기반 군집수 결정
def get_optimal_clusters_with_gemini(embeddings, sample_texts, model):
    prompt = f'''[한국어 지시사항]
샘플 텍스트: {sample_texts[:2000]}
최적 클러스터 수 1개 추천 (10~200):
{{"optimal_clusters": 정수}}'''
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text.strip())
            return result["optimal_clusters"] if 10 <= result["optimal_clusters"] <= 200 else None
        except Exception:
            time.sleep(1)
    return None

# 5. 클러스터 키워드 추출
def gemini_cluster_keywords(cluster_id, texts, model):
    prompt = f'''[한국어 지시사항]
법안 샘플: {texts[:2000]}
4개 키워드 추출:
{{"keywords": ["키워드1", "키워드2", "키워드3", "키워드4"]}}'''
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            return json.loads(response.text.strip())["keywords"][:4]
        except Exception:
            time.sleep(1)
    return ["기타"]

if __name__ == '__main__':
    # 초기화
    GEMINI_API_KEY = "여기에_실제_API키_입력"
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # 데이터 로드
    file_path = Path(r"C:/Users/1-02/Desktop/DAMF2/laws-radar/geovote/data/bill_filtered_final.csv")
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 1. 전처리
    print("🔄 텍스트 전처리 중...")
    df['content'] = parallel_gemini_tokenize_and_filter(df['content'].tolist(), model, 4)

    # 2. 임베딩
    print("🔄 임베딩 생성 중...")
    embeddings = get_gemini_embeddings(df['content'].tolist())
    if not embeddings:
        print("❌ 임베딩 실패")
        sys.exit(1)

    # 3. 군집수 결정
    print("🔎 군집수 분석 중...")
    n_clusters = get_optimal_clusters_with_gemini(embeddings, ' '.join(df.sample(100)['content']), model)
    if not n_clusters:
        print("⚠️ LDA 방식으로 전환")
        vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=(1,3), max_features=5000)
        X = vectorizer.fit_transform(df['content'])
        texts = [doc.split() for doc in df['content']]
        n_clusters = find_optimal_n_topics_lda_fast(X, texts, Dictionary(texts), [], vectorizer)
    print(f"✅ 최종 군집수: {n_clusters}")

    # 4. 클러스터링
    print("🔄 클러스터링 중...")
    df['topic'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(np.array(embeddings))

    # 5. 키워드 추출
    print("🔄 키워드 추출 중...")
    cluster_texts = df.groupby('topic')['content'].apply(lambda x: ' '.join(x.sample(min(10, len(x)))))
    with ThreadPoolExecutor(4) as executor:
        futures = {executor.submit(gemini_cluster_keywords, cid, txt, model): cid 
                  for cid, txt in cluster_texts.items()}
        topic_labels = {futures[f]: f.result() for f in tqdm(as_completed(futures), total=len(futures))}
    df['topic_label'] = df['topic'].map(lambda x: ', '.join(topic_labels.get(x, ['기타'])))

    # 6. 저장
    output_path = Path('data/bill_gemini_clustering.csv')
    df[['age', 'title', 'bill_id', 'bill_number', 'content', 'topic', 'topic_label']].to_csv(
        output_path, index=False, encoding='utf-8-sig'
    )
    print(f"✅ 저장 완료: {output_path}")
