import os
import pandas as pd

def locate_project_root():
    """settings.py 위치 기반 프로젝트 루트 탐색"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if 'settings.py' in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("프로젝트 루트 탐색 실패: settings.py 파일을 찾을 수 없습니다.")
        current_dir = parent_dir

PROJECT_ROOT = locate_project_root()
INPUT_PATHS = {
    'billview': os.path.join(PROJECT_ROOT, 'billview', 'data', 'missing_bill.csv'),
    'geovote': os.path.join(PROJECT_ROOT, 'geovote', 'data', 'bill.csv')
}
OUTPUT_DIRS = {
    'merged': os.path.join(PROJECT_ROOT, 'billview', 'merged_output'),
    'age': os.path.join(PROJECT_ROOT, 'age_classification')
}

def validate_paths():
    """파일 시스템 검증"""
    missing = [k for k,v in INPUT_PATHS.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"누락된 입력 파일: {', '.join(missing)}")
    for d in OUTPUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

def merge_datasets():
    """데이터 병합 핵심 로직"""
    dfs = []
    required_columns = ['age', 'title', 'bill_id', 'bill_number', 'content']
    
    for src, path in INPUT_PATHS.items():
        try:
            df = pd.read_csv(path, encoding='utf-8')
            
            # 필수 컬럼 검증
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise KeyError(f"필수 컬럼 누락: {', '.join(missing_cols)}")
                
            dfs.append(df[required_columns])
            
        except Exception as e:
            raise RuntimeError(f"{src} 데이터 처리 실패: {str(e)}")
    
    merged = pd.concat(dfs, ignore_index=True)
    merged_path = os.path.join(OUTPUT_DIRS['merged'], 'consolidated_data.csv')
    merged.to_csv(merged_path, index=False, encoding='utf-8-sig')
    return merged

def classify_age_data(df):
    """연령 분류 처리기"""
    if 'age' not in df.columns:
        print("경고: 연령 분류를 수행할 수 없습니다.")
        return
    
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # 연령 그룹 분류
    bins = [0, 19, 29, 39, 49, 59, 69, 150]
    labels = ['10대', '20대', '30대', '40대', '50대', '60대', '70대 이상']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    # 그룹별 저장
    for group in labels:
        group_df = df[df['age_group'] == group]
        output_path = os.path.join(OUTPUT_DIRS['age'], f"{group}_data.csv")
        group_df.to_csv(output_path, index=False, encoding='utf-8-sig')

def analyze_missing_content(df):
    """Content 누락 분석"""
    if 'content' not in df.columns:
        print("경고: Content 컬럼이 존재하지 않습니다.")
        return
    
    # NaN과 빈 문자열 모두 카운팅
    missing_count = df['content'].isna().sum()
    empty_count = (df['content'] == '').sum()
    
    print("\n=== Content 누락 분석 ===")
    print(f"NaN 값 개수: {missing_count}건")
    print(f"빈 문자열 개수: {empty_count}건")
    print(f"총 누락 레코드 수: {missing_count + empty_count}건")

def main():
    print("=== 데이터 병합 시스템 시작 ===")
    print(f"인식된 프로젝트 루트: {PROJECT_ROOT}")
    
    try:
        validate_paths()
        merged_data = merge_datasets()
        
        # 누락 content 분석
        analyze_missing_content(merged_data)
        
        # 연령 분류
        classify_age_data(merged_data)
        
        print("\n=== 처리 결과 ===")
        print(f"총 병합 레코드: {len(merged_data):,}건")
        print(f"저장 위치:")
        print(f"- 병합 파일: {os.path.join(OUTPUT_DIRS['merged'], 'consolidated_data.csv')}")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return

if __name__ == "__main__":
    main()
