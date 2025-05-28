import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # TensorFlow 경고 제거

import pandas as pd
import numpy as np
import re
import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from kiwipiepy import Kiwi
from sklearn.cluster import KMeans

class KoBERTClustering:
    def __init__(self, device=None):
        # 디바이스 설정 (GPU 우선)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # KoBERT 모델 및 토크나이저 초기화
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.model = BertModel.from_pretrained('skt/kobert-base-v1').to(self.device)
        self.kiwi = Kiwi()
        self.stopwords = self.get_korean_stopwords()
    
    def get_korean_stopwords(self):
        """한국어 불용어 리스트"""
        return {
            '이', '그', '저', '것', '수', '등', '들', '및', '또는', '또한', 
            '년', '월', '일', '때', '후', '전', '중', '때문', '통해', '대해',
            '의', '가', '에', '를', '은', '는', '으로', '와', '과', '에서', '한', '하다', '있다', '이다'
        }

    def preprocess_text(self, text):
        """텍스트 전처리: 특수문자 제거, Kiwi로 형태소 분석, 불용어 제거, 2글자 이상만"""
        if pd.isnull(text):
            return ""
        text = re.sub(r'[^가-힣\s]', '', str(text))
        tokens = [token.form for token in self.kiwi.tokenize(text)
                  if token.form not in self.stopwords and len(token.form) >= 2]
        return ' '.join(tokens)

    def get_embeddings(self, texts, batch_size=16, max_length=128):
        """KoBERT 임베딩 (배치 처리, [CLS] 토큰 사용)"""
        self.model.eval()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
        return np.vstack(all_embeddings)

    def cluster_documents(self, file_path, n_clusters=5):
        """전체 클러스터링 파이프라인"""
        # 1. 데이터 로드
        df = pd.read_csv(file_path, encoding='utf-8')
        if 'content' not in df.columns:
            raise ValueError("'content' 컬럼이 존재하지 않습니다.")

        # 2. 텍스트 전처리
        print("텍스트 전처리 중...")
        df['cleaned_text'] = df['content'].apply(self.preprocess_text)
        print("전처리 샘플:", df['cleaned_text'].head(3).tolist())

        # 3. 임베딩 생성
        print("임베딩 생성 중...")
        embeddings = self.get_embeddings(df['cleaned_text'].tolist())
        print(f"임베딩 shape: {embeddings.shape}")

        # 4. K-means 클러스터링
        print("클러스터링 수행 중...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(embeddings)

        # 5. 결과 저장
        df.to_csv("clustered_results.csv", index=False, encoding='utf-8-sig')
        print("분석 완료! 결과 파일 저장됨.")
        print(df[['age', 'title', 'bill_id', 'bill_number', 'cluster']].head())
        return df

if __name__ == "__main__":
    # 파일 경로와 클러스터 수 설정
    file_path = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\geovote\data\bill_filtered_final.csv"
    n_clusters = 5

    analyzer = KoBERTClustering()
    result_df = analyzer.cluster_documents(file_path, n_clusters)
