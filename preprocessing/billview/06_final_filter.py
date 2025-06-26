# bill_updated.csv에서 결측값 제거

import pandas as pd
import os
from pathlib import Path

class CSVContentCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.filtered_df = None
        
    def load_data(self):
        """CSV 파일을 로드합니다."""
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
            
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.file_path, encoding=encoding)
                    print(f"파일을 성공적으로 로드했습니다 (인코딩: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("지원되는 인코딩으로 파일을 읽을 수 없습니다.")
                
            return True
            
        except Exception as e:
            print(f"파일 로딩 오류: {str(e)}")
            return False
    
    def analyze_content_column(self):
        """content 컬럼을 분석합니다."""
        if self.df is None:
            print("먼저 데이터를 로드해주세요.")
            return
            
        if 'content' not in self.df.columns:
            print("'content' 컬럼이 존재하지 않습니다.")
            print(f"사용 가능한 컬럼: {list(self.df.columns)}")
            return
        
        total_rows = len(self.df)
        
        # 다양한 유형의 빈 값 분석
        na_count = self.df['content'].isna().sum()
        empty_string_count = (self.df['content'].astype(str).str.strip() == '').sum()
        nan_string_count = (self.df['content'].astype(str).str.strip() == 'nan').sum()
        none_string_count = (self.df['content'].astype(str).str.strip().str.lower() == 'none').sum()
        
        print("=== Content 컬럼 분석 ===")
        print(f"전체 행 수: {total_rows}")
        print(f"NaN 값: {na_count} 개")
        print(f"빈 문자열: {empty_string_count} 개")
        print(f"'nan' 문자열: {nan_string_count} 개")
        print(f"'none' 문자열: {none_string_count} 개")
        
        # 유효한 content 행 수
        valid_content = self.df[
            self.df['content'].notna() & 
            (self.df['content'].astype(str).str.strip() != '') &
            (self.df['content'].astype(str).str.strip() != 'nan') &
            (self.df['content'].astype(str).str.strip().str.lower() != 'none')
        ]
        
        print(f"유효한 content를 가진 행: {len(valid_content)} 개")
        print(f"제거될 행: {total_rows - len(valid_content)} 개")
        
        return len(valid_content)
    
    def clean_content(self, save_backup=True):
        """content가 비어있는 행들을 제거합니다."""
        if self.df is None:
            print("먼저 데이터를 로드해주세요.")
            return False
        
        # 백업 생성
        if save_backup:
            backup_path = self.file_path.with_suffix('.backup.csv')
            self.df.to_csv(backup_path, index=False, encoding='utf-8-sig')
            print(f"백업 파일 생성: {backup_path}")
        
        # content 필터링
        self.filtered_df = self.df[
            self.df['content'].notna() & 
            (self.df['content'].astype(str).str.strip() != '') &
            (self.df['content'].astype(str).str.strip() != 'nan') &
            (self.df['content'].astype(str).str.strip().str.lower() != 'none')
        ]
        
        print(f"필터링 완료: {len(self.df)} → {len(self.filtered_df)} 행")
        return True
    
    def save_cleaned_data(self, output_path=None):
        """정리된 데이터를 저장합니다."""
        if self.filtered_df is None:
            print("먼저 데이터를 정리해주세요.")
            return False
        
        if output_path is None:
            output_path = self.file_path.with_suffix('.filtered.csv')
        
        self.filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"정리된 데이터 저장 완료: {output_path}")
        return True
    
    def get_sample_data(self, n=5):
        """샘플 데이터를 반환합니다."""
        if self.filtered_df is not None:
            print("=== 정리된 데이터 샘플 ===")
            return self.filtered_df.head(n)
        elif self.df is not None:
            print("=== 원본 데이터 샘플 ===")
            return self.df.head(n)
        else:
            print("로드된 데이터가 없습니다.")
            return None

# 사용 예시
def main():
    file_path = r'C:\Users\1-02\Desktop\DAMF2\laws-radar\billview\filter\bill_updated.csv'
    
    # CSV 정리기 생성
    cleaner = CSVContentCleaner(file_path)
    
    # 데이터 로드
    if not cleaner.load_data():
        return
    
    # content 컬럼 분석
    cleaner.analyze_content_column()
    
    # 데이터 정리
    if cleaner.clean_content():
        # 정리된 데이터 저장
        cleaner.save_cleaned_data()
        
        # 샘플 데이터 출력
        sample = cleaner.get_sample_data()
        if sample is not None:
            print(sample)

if __name__ == "__main__":
    main()
