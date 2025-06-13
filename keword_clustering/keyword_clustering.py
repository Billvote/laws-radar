# -*- coding: utf-8 -*-
import sys
import gc
import pickle
import asyncio
import aiohttp
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import google.generativeai as genai
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

tqdm.pandas()

# 기존 필터 설정 유지 + 법률 특화 용어 추가
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
    # 법률 특화 용어 추가
    '송환대기실', '입국불허', '밀입국', '출입국관리', '외국인관서', '운수업자', '항공사운영협의회',
    '국민안전', '위협요소', '사후관리', '보안안전', '감사결과', '법적근거', '특별사유'
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

# 법률 특화 불용어 추가
legal_specific_stopwords = frozenset({
    '있는', '있음', '되는', '되도록', '하는', '하도록', '지적한', '마련할', '부여함에',
    '상황임', '있게', '함으로써', '하고자', '경우에는', '있는지', '있다면', '하여야',
    '하여서는', '아니하는', '아니한', '아니되는', '아니된', '있으므로', '있어서',
    '가능한', '필요한', '적절한', '효과적인', '지속적으로', '전문적인', '체계적인'
})

preserve_terms = frozenset({
    '법률', '법안', '입법', '개정', '제정', '시행', '공포', '폐지', '조례', '규정', '조항', '의결',
    '감사원', '국민', '안전', '위협', '요소', '대응', '실태', '공항', '보안', '분야', 
    '감사', '결과', '입국', '불허', '사후', '미흡', '자유', '이동', '일부', '밀입국',
    '시도', '발생', '일반인', '분리', '구별', '출국', '송환', '대기실', '통제', '마련',
    '지적', '허가', '외국인', '선박', '운수업', '의무', '부여', '민간', '항공사',
    '협의회', '본국', '임시', '문제점', '제기', '지방', '관서', '효과', '일정', '장소',
    '제공', '요청', '특별', '사유', '협조', '신설'
})

excluded_bigrams = frozenset({'교육 실시', '징역 벌금', '수립 시행', '운영 관리'})

# Gemini API 키 설정
GEMINI_API_KEY = "AIzaSyA8M00iSzCK1Lvc5YfxamYgQf-Lh4xh5R0"
genai.configure(api_key=GEMINI_API_KEY)

# ===== 법률 문서 특화 전처리 함수들 =====

def enhanced_law_pattern_removal(text):
    """향상된 법률 구조 패턴 제거"""
    if pd.isna(text) or text is None or not isinstance(text, str):
        return ""
    
    try:
        patterns = [
            # 기본 법조문 패턴
            r'제?\d+조의?\d*(?:제?\d+항)?(?:제?\d+호)?',
            r'안\s*제?\d+조의?\d*(?:제?\d+항)?(?:제?\d+호)?',
            
            # 날짜 및 수량 패턴
            r'\d+년\s*\d+월\s*\d+일?',
            r'\d+만\s*\d+천?\s*\d+명?',
            r'\d+%',
            
            # 법률 문서 특수 패턴
            r'\([^)]*법[^)]*\)',  # 법률명 괄호
            r'\([^)]*년[^)]*\)',  # 년도 괄호
            r'\'[^\']*\'',        # 작은따옴표 내용
            r'"[^"]*"',           # 큰따옴표 내용
            r'？',                # 특수 물음표
            
            # 불필요한 조사 및 어미
            r'\b(?:누구나|지니고|유사한|기준|약)\b',
            r'신설', r'정비', r'조정', r'인용조문', r'정비\s*\(.*?\)', r'안'
        ]
        
        combined = '|'.join(patterns)
        text = re.sub(combined, ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
        
    except Exception as e:
        print(f"법령 구조 패턴 제거 오류: {str(e)}")
        return text

def compound_noun_handler(text):
    """복합 명사 및 특수 문자 처리"""
    if not text:
        return ""
    
    try:
        # 1. 특수 문자 처리
        text = re.sub(r'·', ' ', text)  # 중점 제거
        text = re.sub(r'？', ' ', text)  # 특수 물음표 제거
        
        # 2. 복합 명사 분리 패턴
        compound_patterns = {
            r'지방출입국·?외국인관서': '지방 출입국 외국인 관서',
            r'항공사운영협의회': '항공사 운영 협의회',
            r'송환대기실': '송환 대기실',
            r'입국불허자': '입국 불허',
            r'밀입국시도': '밀입국 시도',
            r'사후관리': '사후 관리',
            r'보안안전': '보안 안전',
            r'법적근거': '법적 근거',
            r'위협요소': '위협 요소'
        }
        
        for pattern, replacement in compound_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # 3. 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    except Exception as e:
        print(f"복합 명사 처리 오류: {str(e)}")
        return text

def context_aware_stopword_filter(text):
    """맥락 인식 계층적 불용어 처리"""
    if not text:
        return ""
    
    try:
        words = text.split()
        filtered_words = []
        
        for i, word in enumerate(words):
            # 길이 필터링
            if len(word) < 2:
                continue
            
            # 숫자만으로 구성된 단어 제거
            if word.isdigit():
                continue
            
            # 기본 불용어 체크
            if word in initial_stopwords:
                continue
            
            # 제외 용어 체크
            if word in initial_excluded_terms:
                continue
            
            # 법률 특화 불용어 체크
            if word in legal_specific_stopwords:
                continue
            
            # 보존 용어는 항상 유지
            if word in preserve_terms:
                filtered_words.append(word)
                continue
            
            # 사용자 정의 법률 용어 유지
            if word in custom_nouns:
                filtered_words.append(word)
                continue
            
            # 2글자 이상 일반 용어
            if len(word) >= 2:
                # 동사/형용사 어미 패턴 제거
                if not re.search(r'(하다|되다|있다|없다|이다|아니다)$', word):
                    filtered_words.append(word)
        
        return ' '.join(filtered_words)
        
    except Exception as e:
        print(f"불용어 필터링 오류: {str(e)}")
        return text

def legal_document_preprocessing_pipeline(text):
    """법률 문서 전용 전처리 파이프라인"""
    if pd.isna(text) or text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    if not text.strip():
        return ""
    
    try:
        # 1단계: 기본 법령 구조 패턴 제거
        text = enhanced_law_pattern_removal(text)
        
        # 2단계: 복합 명사 및 특수 문자 처리
        text = compound_noun_handler(text)
        
        # 3단계: 맥락 인식 불용어 필터링
        text = context_aware_stopword_filter(text)
        
        # 4단계: 최종 정규화
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text if text else ""
        
    except Exception as e:
        print(f"법률 문서 전처리 오류: {str(e)}")
        return text

def improved_content_preprocessing(df):
    """개선된 content 전처리 파이프라인"""
    print("📊 법률 문서 특화 전처리 시작...")
    
    # 1. 데이터 타입 분포 확인
    print("데이터 타입 분포:")
    print(df['content'].apply(lambda x: type(x)).value_counts())
    
    # 2. 결측값 처리
    print("🔄 결측값 처리 중...")
    df['content'] = df['content'].fillna('')
    
    # 3. 데이터 타입 통일
    print("🔄 데이터 타입 통일 중...")
    df['content'] = df['content'].astype(str)
    
    # 4. 빈 값 정리
    print("🔄 빈 값 정리 중...")
    df['content'] = df['content'].replace(['nan', 'None'], '')
    
    # 5. 법률 문서 특화 전처리 적용
    print("🔄 법률 문서 특화 전처리 적용 중...")
    df['content'] = df['content'].apply(legal_document_preprocessing_pipeline)
    
    # 6. 결과 검증
    empty_count = (df['content'] == '').sum()
    print(f"✅ 전처리 완료: {len(df)}개 중 {empty_count}개 빈 텍스트")
    
    # 7. 샘플 결과 출력
    print("\n📋 법률 특화 전처리 결과 샘플:")
    for i in range(min(3, len(df))):
        if df.iloc[i]['content']:
            print(f"   {i+1}. {df.iloc[i]['content'][:150]}...")
    
    return df

# ===== 2단계: Gemini 원본 기반 클러스터링 =====

def gemini_clustering_from_original(original_texts, titles, model):
    """Gemini가 원본 텍스트를 직접 분석하여 클러스터링 수행"""
    print(f"🤖 Gemini 원본 텍스트 기반 클러스터링 시작...")
    
    # 문서별 주제 분류
    def classify_document(idx_doc):
        idx, doc, title = idx_doc
        prompt = f"""
다음 법안의 원본 내용을 분석하여 주제 카테고리를 분류해주세요.

법안 제목: {title}
법안 원본 내용: {doc[:3000]}

다음 주제 카테고리 중 하나를 선택하거나 새로운 카테고리를 제안하세요:
- 교육정책
- 보건의료  
- 경제금융
- 환경에너지
- 사회복지
- 국방안보
- 법무사법
- 행정안전
- 과학기술
- 문화체육
- 농림수산
- 국토교통
- 외교통일
- 디지털정보통신
- 출입국관리
- 기타

다음 형식으로 응답해주세요:
{{
  "category": "주제_카테고리",
  "subcategory": "세부_분류",
  "confidence": 0.9
}}
"""
        
        try:
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return idx, result
        except Exception as e:
            print(f"문서 {idx} 분류 실패: {str(e)}")
        
        return idx, {"category": "기타", "subcategory": "일반", "confidence": 0.5}
    
    # 병렬 처리로 문서 분류
    document_classifications = {}
    data_with_index = [(i, original_texts[i], titles[i]) for i in range(len(original_texts))]
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(classify_document, item) for item in data_with_index]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="원본 텍스트 문서 분류"):
            idx, classification = future.result()
            document_classifications[idx] = classification
    
    # 카테고리별 그룹화
    category_groups = defaultdict(list)
    for idx, classification in document_classifications.items():
        category = classification["category"]
        subcategory = classification.get("subcategory", "일반")
        key = f"{category}_{subcategory}"
        category_groups[key].append(idx)
    
    # 클러스터 ID 할당
    final_clusters = {}
    cluster_id = 0
    
    for category_key, doc_indices in category_groups.items():
        if len(doc_indices) <= 5:
            # 소규모 그룹은 단일 클러스터
            for doc_idx in doc_indices:
                final_clusters[doc_idx] = cluster_id
            cluster_id += 1
        else:
            # 대규모 그룹은 추가 세분화
            subclusters = gemini_subcluster_documents(
                [(idx, original_texts[idx], titles[idx]) for idx in doc_indices], 
                model, 
                category_key
            )
            
            for subcluster_docs in subclusters:
                for doc_idx in subcluster_docs:
                    final_clusters[doc_idx] = cluster_id
                cluster_id += 1
    
    return final_clusters

def gemini_subcluster_documents(docs_data, model, category):
    """카테고리 내 세부 클러스터링"""
    if len(docs_data) <= 10:
        return [[doc[0] for doc in docs_data]]
    
    prompt = f"""
다음 {category} 카테고리의 법안들을 유사한 세부 주제별로 2-4개 그룹으로 나누어주세요.

법안 목록:
"""
    
    for i, (idx, text, title) in enumerate(docs_data[:15]):
        prompt += f"{i+1}. {title[:80]}...\n"
    
    prompt += """
각 그룹에 속하는 법안 번호들을 다음 형식으로 응답해주세요:

{
  "groups": [
    {
      "name": "그룹명1",
      "bills": [1, 3, 5]
    },
    {
      "name": "그룹명2", 
      "bills": [2, 4, 6]
    }
  ]
}
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            subclusters = []
            used_indices = set()
            
            for group in result.get("groups", []):
                group_doc_indices = []
                for bill_num in group.get("bills", []):
                    if 1 <= bill_num <= len(docs_data) and (bill_num - 1) not in used_indices:
                        doc_idx = docs_data[bill_num - 1][0]
                        group_doc_indices.append(doc_idx)
                        used_indices.add(bill_num - 1)
                
                if group_doc_indices:
                    subclusters.append(group_doc_indices)
            
            # 미분류 문서들 처리
            remaining = [docs_data[i][0] for i in range(len(docs_data)) if i not in used_indices]
            if remaining:
                subclusters.append(remaining)
            
            return subclusters if subclusters else [[doc[0] for doc in docs_data]]
            
    except Exception as e:
        print(f"세부 클러스터링 실패: {str(e)}")
    
    return [[doc[0] for doc in docs_data]]

# ===== 3단계: Gemini 원본 기반 키워드 추출 =====

def gemini_extract_cluster_keywords(cluster_docs, original_texts, titles, model):
    """클러스터별 원본 텍스트 기반 키워드 추출"""
    # 클러스터 내 대표 문서들 선택
    sample_indices = cluster_docs[:3]  # 최대 3개 문서
    combined_text = ' '.join([original_texts[i] for i in sample_indices])
    representative_title = titles[sample_indices[0]] if sample_indices else "법안"
    
    prompt = f"""
다음 클러스터에 속한 법안들의 원본 내용을 분석하여 공통된 핵심 키워드 4개를 추출해주세요.

대표 제목: {representative_title}
클러스터 원본 내용: {combined_text[:4000]}

추출 규칙:
1. 클러스터 내 모든 문서에 공통으로 나타나는 용어 우선
2. 법률 전문용어를 우선적으로 선택
3. 원본 내용에 실제로 등장하는 용어만 사용
4. 클러스터의 주제를 가장 잘 대표하는 키워드 선택

반드시 다음 JSON 배열 형식으로만 응답하세요:
["키워드1", "키워드2", "키워드3", "키워드4"]
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        match = re.search(r'\["[^"]*"(?:\s*,\s*"[^"]*")*\]', result_text)
        if match:
            keywords = json.loads(match.group())
            return keywords[:4]
    except Exception as e:
        print(f"클러스터 키워드 추출 오류: {str(e)}")
    
    return ["법안", "개정", "정책", "시행"]

def gemini_extract_single_keywords(original_text, title, model):
    """단일 문서 원본 텍스트 기반 키워드 추출"""
    prompt = f"""
다음 법안의 원본 내용을 분석하여 핵심 키워드 4개를 추출해주세요.

법안 제목: {title}
법안 원본 내용: {original_text[:3000]}

추출 규칙:
1. 반드시 원본 내용에 실제로 등장하는 용어만 사용
2. 법률 전문용어를 우선적으로 선택
3. 법안의 핵심 목적과 직접 연관된 키워드 선택
4. 추상적 개념보다 구체적 용어 우선

반드시 다음 JSON 배열 형식으로만 응답하세요:
["키워드1", "키워드2", "키워드3", "키워드4"]
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        match = re.search(r'\["[^"]*"(?:\s*,\s*"[^"]*")*\]', result_text)
        if match:
            keywords = json.loads(match.group())
            return keywords[:4]
    except Exception as e:
        print(f"단일 문서 키워드 추출 오류: {str(e)}")
    
    return ["법안", "개정", "정책", "시행"]

# ===== 메인 실행 함수 =====

def legal_specialized_processing_system():
    """법률 문서 특화 처리 시스템"""
    print("🚀 법률 문서 특화 처리 시스템 시작")
    start_time = time.time()
    
    # Gemini 모델 생성 (클러스터링과 키워드 추출용)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # 데이터 로드
    print("📊 데이터 로드 중...")
    file_path = Path(r"C:/Users/1-02/Desktop/DAMF2/laws-radar/geovote/data/bill_filtered_final.csv")
    
    dtype_spec = {
        'age': 'int16',
        'bill_id': 'category'
    }
    
    df = pd.read_csv(file_path, dtype=dtype_spec, encoding='utf-8-sig')
    print(f"📊 원본 데이터: {len(df)}개 의안")

    # 원본 텍스트 별도 보관 (클러스터링과 키워드 추출용)
    df['original_content'] = df['content'].copy()

    # 1단계: 법률 문서 특화 전처리 파이프라인
    print("🔄 법률 문서 특화 전처리 파이프라인 수행 중...")
    df = improved_content_preprocessing(df)

    # 전처리 결과 검증
    print("\n🔍 법률 특화 전처리 결과 검증:")
    processed_count = (df['content'] != '').sum()
    print(f"   - 처리된 문서: {processed_count}개")
    print(f"   - 빈 문서: {len(df) - processed_count}개")
    
    # 샘플 확인을 위한 디버깅 출력
    if processed_count > 0:
        sample_idx = df[df['content'] != ''].index[0]
        print(f"   - 샘플 결과: {df.iloc[sample_idx]['content'][:200]}...")

    # 2단계: Gemini 원본 기반 클러스터링
    print("🤖 Gemini 원본 텍스트 기반 클러스터링 수행 중...")
    clusters = gemini_clustering_from_original(
        df['original_content'].tolist(),
        df['title'].tolist(),
        model
    )
    
    # 클러스터 결과를 데이터프레임에 적용
    df['topic'] = df.index.map(lambda x: clusters.get(x, -1))

    # 3단계: Gemini 원본 기반 키워드 추출
    print("🔄 Gemini 원본 텍스트 기반 키워드 추출 중...")
    topic_labels = {}
    
    # 클러스터별 키워드 추출
    unique_topics = df[df['topic'] != -1]['topic'].unique()
    
    def extract_cluster_keywords(cid):
        cluster_docs = df[df['topic'] == cid].index.tolist()
        
        try:
            keywords = gemini_extract_cluster_keywords(
                cluster_docs,
                df['original_content'].tolist(),
                df['title'].tolist(),
                model
            )
            return cid, keywords
            
        except Exception as e:
            print(f"클러스터 {cid} 키워드 추출 실패: {str(e)}")
            return cid, ["법안", "개정", "정책", "시행"]
    
    # 병렬 처리로 클러스터 키워드 추출
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(extract_cluster_keywords, cid) for cid in unique_topics]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="클러스터 키워드 추출"):
            cid, keywords = future.result()
            topic_labels[cid] = keywords

    # 단일 문서 키워드 추출
    single_docs = df[df['topic'] == -1]
    print(f"🔄 단일 문서 키워드 추출 중... ({len(single_docs)}개)")
    
    def extract_single_doc_keywords(idx_row):
        idx, row = idx_row
        try:
            keywords = gemini_extract_single_keywords(
                row['original_content'], 
                row['title'], 
                model
            )
            unique_topic_id = -1 * (idx + 2)
            return idx, unique_topic_id, keywords
        except Exception as e:
            print(f"단일 문서 키워드 추출 실패: {str(e)}")
            unique_topic_id = -1 * (idx + 2)
            return idx, unique_topic_id, ["법안", "개정", "정책", "시행"]
    
    if len(single_docs) > 0:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(extract_single_doc_keywords, (idx, row)) 
                      for idx, row in single_docs.iterrows()]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="단일 문서 처리"):
                idx, unique_topic_id, keywords = future.result()
                topic_labels[unique_topic_id] = keywords
                df.at[idx, 'topic'] = unique_topic_id

    # topic_label 컬럼 생성
    df['topic_label'] = df['topic'].map(lambda x: ', '.join(topic_labels.get(x, [])))
    
    # 빈 키워드 라벨 처리
    empty_labels = df[df['topic_label'] == '']
    if len(empty_labels) > 0:
        print(f"⚠️ 빈 키워드 라벨 {len(empty_labels)}개 발견 - 기본 키워드로 대체")
        df.loc[df['topic_label'] == '', 'topic_label'] = '법안, 개정, 정책, 시행'

    # original_content 컬럼 제거
    df.drop('original_content', axis=1, inplace=True)

    print(f"✅ 최종 처리된 의안: {len(df)}개")
    print(f"✅ 생성된 클러스터: {len(set(df['topic']))}개")

    # 결과 저장
    output_path = Path('data/bill_legal_specialized_processing.csv')
    output_path.parent.mkdir(exist_ok=True)
    output_columns = ['age', 'title', 'bill_id', 'bill_number', 'content', 'topic', 'topic_label']
    df[output_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 저장 완료: {output_path}")
    
    # 처리 시간 계산
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"⏱️ 총 처리 시간: {processing_time:.2f}초 ({processing_time/60:.2f}분)")
    
    # 결과 요약
    print("\n📈 법률 문서 특화 처리 결과 요약:")
    topic_summary = df.groupby('topic').agg({
        'title': 'count',
        'topic_label': 'first'
    }).rename(columns={'title': 'document_count'}).sort_values('document_count', ascending=False)
    
    print(f"   - 총 클러스터 수: {len(topic_summary)}개")
    print(f"   - 총 문서 수: {len(df)}개")
    
    print(f"\n🔍 주요 클러스터별 키워드 (법률 특화):")
    for i, (topic_id, row) in enumerate(topic_summary.head(10).iterrows()):
        keywords = row['topic_label']
        print(f"   {i+1}. [{row['document_count']}개 문서] {keywords}")
    
    print(f"\n🎯 법률 문서 특화 처리 방식:")
    print(f"   - ✅ content 처리: 법률 문서 특화 전처리 파이프라인")
    print(f"   - ✅ 복합 명사 분리: '지방출입국·외국인관서' → '지방 출입국 외국인 관서'")
    print(f"   - ✅ 법률 용어 보존: 50개 이상 법률 전문용어 보존")
    print(f"   - ✅ 맥락 인식 필터링: 3단계 계층적 불용어 제거")
    print(f"   - ✅ 클러스터링: Gemini가 원본 텍스트 기반 수행")
    print(f"   - ✅ 키워드 추출: Gemini가 원본 텍스트 기반 수행")
    
    return df, topic_labels

if __name__ == '__main__':
    # 디렉토리 생성
    Path("data").mkdir(exist_ok=True)
    
    # 법률 문서 특화 처리 시스템 실행
    df_result, topic_labels_result = legal_specialized_processing_system()
