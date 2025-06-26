import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings

import pandas as pd
import os

from _01_save_bill_ids.save_bill_ids import fetch_and_save_bill_ids
from _02_result_vote.result_vote_crawling import collect_vote_data
from _03_bill_crawling.bill_summary_crawling import crawl_summaries
from _04_keyword_clustering.keyword_gemini import legal_preprocessing_only
# from _05_generate_label.label_final import add_label_column
from _05_generate_summary.summary_of_content import initialize_system, generate_summary, generate_summaries_parallel
from _06_generate_cardnews.card_news_main import CardNewsConverter

def run_all(eraco: str):
    # gemini API 클라이언트 초기화
    gemini_client = initialize_system()
    if gemini_client is None:
        print("Gemini API 초기화 실패, 요약 작업을 건너뜁니다.")
        return
    
    # 1. 의안 ID 수집-------------------------------
    print(f"STEP 1: {eraco}대 의안 ID 수집 중")
    eraco_for_ids = f'제{eraco}대'   # 의안 ID용 (한글 포함)
    eraco_for_vote = eraco          # 표결 데이터용 (숫자만)
    bill_ids = fetch_and_save_bill_ids(eraco_for_ids)

    # 2. 표결 데이터 수집----------------------------
    print(f"STEP 2: {eraco}대 표결 데이터 수집 중")
    # bill_ids = pd.read_csv(id_csv_path)['bill_id'].tolist()

    df_votes = collect_vote_data(bill_ids, eraco_for_vote) # 테스트용
    print(f"▶ 표결 데이터 크기: {df_votes.shape}")
    print(df_votes.head(3))

    # 3. 제안이유/주요내용 파싱-----------------------
    print(f"STEP 3: {eraco}대 제안이유/주요내용 크롤링 중")
    urls = df_votes['BILL_URL'].dropna().unique().tolist()
    crawl_results = crawl_summaries(urls)
    result_df = pd.DataFrame(crawl_results)
    print(f"▶ 크롤링된 summary 개수: {len(result_df)}")
    print(result_df.head(3))
    merged_df = result_df.groupby('url', as_index=False)['content'].agg(
        lambda x: ' / '.join([str(i) for i in x if pd.notna(i)])
    )
    print(f"▶ 병합 전 summary 그룹화 결과 개수: {merged_df.shape}")

    # 3.5. df_votes와 summary 병합 및 컬럼 정리------
    print(f"STEP 3.5: 제안이유와 표결 데이터 병합 중")
    df_votes_merged = pd.merge(df_votes, merged_df, how='left', left_on='BILL_URL', right_on='url')
    print(f"▶ 병합 후 데이터 크기: {df_votes_merged.shape}")
    print("▶ 병합된 데이터 내 summary null 개수:", df_votes_merged['content'].isna().sum())
    print(df_votes_merged[['BILL_NAME', 'BILL_URL', 'content']].sample(3))
    
    # 결측치가 있는 행 제거 (content 기준)
    df_votes_merged = df_votes_merged.dropna(subset=['content'])
    print(f"▶ content 결측치 제거 후 데이터 크기: {df_votes_merged.shape}")
    
    df_votes_merged.drop(columns=['url'], inplace=True)
    df_votes_merged.rename(columns={
            'AGE': 'age',
            'MONA_CD': 'member_id',
            'BILL_NAME': 'title',
            'BILL_NO': 'bill_number',
            'BILL_ID': 'bill_id',
            'RESULT_VOTE_MOD': 'result',
            'VOTE_DATE': 'date',
            'BILL_URL': 'url',
            }, inplace=True)
    # date 형식 변환
    df_votes_merged['date'] = pd.to_datetime(
        df_votes_merged['date'].astype(str).str[:8],
        format='%Y%m%d',
        errors='coerce'
        ).dt.date

    # 유니크한 법만 뽑아서 작업: 형태소 분석, summary, cardnews
    # 4. 법안별 중복 제거 후 전처리 및 요약/카드뉴스 생성
    print("STEP 4: 법안별 중복 제거 및 텍스트 처리 시작")
    unique_bills = df_votes_merged[['bill_id', 'content']].drop_duplicates(subset=['bill_id']).reset_index(drop=True)
    unique_bills = unique_bills.head(10) # 테스트용

    # 4-1. 형태소 분석
    unique_bills = legal_preprocessing_only(unique_bills)
    unique_bills.drop(columns=['original_content'], inplace=True)

    # 4-2. 한줄 요약 생성
    print("STEP 5: 한줄 요약 생성 중")
    from tqdm import tqdm
    tqdm.pandas()
    unique_bills = generate_summaries_parallel(unique_bills, gemini_client, max_workers=5)
    print("▶ 한줄 요약 생성 완료")

    # 4-3. 카드뉴스 문구 생성
    print("STEP 6: 카드뉴스 문구 생성 중")
    try:
        card_converter = CardNewsConverter()
        unique_bills['card_news_content'] = [
            card_converter.convert_single(text)
            for text in unique_bills['summary']
        ]
        print("▶ 카드뉴스 문구 생성 완료")
    except Exception as e:
        print(f"카드뉴스 생성 중 오류: {e}")

    # 5. 표결 데이터에 법안별 처리 결과 병합
    print("STEP 7: 표결 데이터에 처리 결과 병합")
    final_df = pd.merge(df_votes_merged, 
                        unique_bills[['bill_id', 'content', 'summary', 'card_news_content']], 
                        on='bill_id', how='left')
    final_df.drop(columns=['content_x'], inplace=True)
    final_df.rename(columns={'content_y': 'cleaned'}, inplace=True)

    print(f"▶ 최종 데이터 크기: {final_df.shape}")
    print(final_df[['bill_id', 'title', 'summary', 'card_news_content']].head(3))

    # 3대 각각 돌린 다음(형태소 분석까지만), 별도 함수에서 3개 대수 merge한 다음에 -> 클러스터링해서 번호+키워드 -> 한줄요약, 카드뉴스 -> 라벨링

    # --------------------------------------------
    # csv 저장
    def save_to_csv(df: pd.DataFrame, eraco: str, file_name: str = None, save_dir: str = None):
        if file_name is None:
            file_name = f'df_{eraco}.csv'
        if save_dir is None:
            save_dir = settings.BASE_DIR / 'run_pipeline' / 'data'
        save_path = save_dir / file_name

        df.to_csv(save_path, index=False, encoding='utf-8-sig')

    save_to_csv(df_votes, eraco_for_vote, '01_all_df_votes.csv')
    save_to_csv(merged_df, eraco_for_vote, '02_all_merged_df.csv')
    save_to_csv(df_votes_merged, eraco_for_vote, '03_all_df_votes_merged.csv')
    save_to_csv(unique_bills, eraco_for_vote, '04_all_unique_bills.csv')
    save_to_csv(final_df, eraco_for_vote, '05_all_final_df.csv')

    print("전체 파이프라인 완료!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용 예시: python run_pipeline.py 22")
        sys.exit(1)

    congress_number = sys.argv[1]

    if not congress_number.isdigit():
        print("오류: 대수는 숫자여야 합니다.")
        sys.exit(1)

    run_all(congress_number)
