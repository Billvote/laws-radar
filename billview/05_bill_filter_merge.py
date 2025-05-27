# bill.csv와 fiterd_merge_comma를 병합
# bill_updated.csv로 저장

import pandas as pd
import os

# 파일 경로 설정
bill_file_path = r"C:/Users/1-02/Desktop/DAMF2/laws-radar/billview/filter/bill.csv"
filter_file_path = r"C:/Users/1-02/Desktop/DAMF2/laws-radar/billview/filter/fiterd_merge_comma.csv"

try:
    # CSV 파일 읽기
    print("CSV 파일을 읽는 중...")
    bill_df = pd.read_csv(bill_file_path, encoding='utf-8')
    filter_df = pd.read_csv(filter_file_path, encoding='utf-8')
    
    print(f"bill.csv 파일 크기: {len(bill_df)}행")
    print(f"filter_1_30.csv 파일 크기: {len(filter_df)}행")
    
    # 데이터프레임 구조 확인
    print("\nbill.csv 컬럼:", bill_df.columns.tolist())
    print("filter_1_30.csv 컬럼:", filter_df.columns.tolist())
    
    # bill_number 컬럼이 존재하는지 확인
    if 'bill_number' not in bill_df.columns:
        print("경고: bill.csv에 'bill_number' 컬럼이 없습니다.")
        print("사용 가능한 컬럼:", bill_df.columns.tolist())
        
    if 'bill_number' not in filter_df.columns:
        print("경고: filter_1_30.csv에 'bill_number' 컬럼이 없습니다.")
        print("사용 가능한 컬럼:", filter_df.columns.tolist())
    
    # content 컬럼이 filter_1_30.csv에 존재하는지 확인
    if 'content' not in filter_df.columns:
        print("경고: filter_1_30.csv에 'content' 컬럼이 없습니다.")
        print("사용 가능한 컬럼:", filter_df.columns.tolist())
        exit(1)
    
    # bill.csv에 content 컬럼이 없으면 생성
    if 'content' not in bill_df.columns:
        bill_df['content'] = ''
        print("bill.csv에 빈 'content' 컬럼을 생성했습니다.")
    
    # 🔽 bill_number 컬럼을 문자열로 통일 (핵심 수정)
    bill_df['bill_number'] = bill_df['bill_number'].astype(str)
    filter_df['bill_number'] = filter_df['bill_number'].astype(str)
    
    # 매칭 전 상태 확인
    before_fill_count = bill_df['content'].notna().sum()
    print(f"\n매칭 전 bill.csv의 content가 채워진 행 수: {before_fill_count}")
    
    # bill_number를 기준으로 매칭하여 content 업데이트
    merged_df = bill_df.merge(
        filter_df[['bill_number', 'content']], 
        on='bill_number', 
        how='left', 
        suffixes=('', '_new')
    )
    
    # content_new 컬럼의 값으로 기존 content 컬럼 업데이트
    bill_df['content'] = merged_df['content_new'].fillna(bill_df['content'])
    
    # 매칭 결과 확인
    after_fill_count = bill_df['content'].notna().sum()
    matched_count = len(filter_df[filter_df['bill_number'].isin(bill_df['bill_number'])])
    
    print(f"매칭 후 bill.csv의 content가 채워진 행 수: {after_fill_count}")
    print(f"새로 채워진 content 수: {after_fill_count - before_fill_count}")
    print(f"filter_1_30.csv에서 매칭된 bill_number 수: {matched_count}")
    
    # 결과를 새 파일로 저장
    output_path = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\billview\filter\bill_updated.csv"
    bill_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n업데이트된 데이터가 '{output_path}'에 저장되었습니다.")
    
    # 매칭되지 않은 bill_number 확인 (선택사항)
    unmatched_bills = bill_df[~bill_df['bill_number'].isin(filter_df['bill_number'])]
    if len(unmatched_bills) > 0:
        print(f"\n매칭되지 않은 bill_number 수: {len(unmatched_bills)}")
        print("매칭되지 않은 bill_number 중 처음 5개:")
        print(unmatched_bills['bill_number'].head().tolist())

except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
    print("파일 경로를 확인해주세요.")
    
except UnicodeDecodeError:
    print("파일 인코딩 문제가 발생했습니다. 다른 인코딩을 시도합니다...")
    try:
        # 다른 인코딩으로 재시도
        bill_df = pd.read_csv(bill_file_path, encoding='cp949')
        filter_df = pd.read_csv(filter_file_path, encoding='cp949')
        print("cp949 인코딩으로 파일을 성공적으로 읽었습니다.")
    except:
        print("인코딩 문제를 해결할 수 없습니다. 파일 인코딩을 확인해주세요.")
        
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")
