from pathlib import Path

# laws-radar로 베이스 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent

# 각 폴더의 data 경로 설정 (루트 디렉토리 기준으로 상대경로 사용)
BILL_SUMMARY_DATA_DIR = BASE_DIR / 'bill_summary' / 'data'
GEO_VISUALIZATION_DATA_DIR = BASE_DIR / 'geo_visualization'
GEOVOTE_DATA_DIR = BASE_DIR / 'geovote' / 'data'
KEYWORD_CLUSTERING_DATA_DIR = BASE_DIR / 'keyword_clustering' / 'data'
MEMBER_INFO_DATA_DIR = BASE_DIR / 'member_info' / 'data'
MERGED_DATA_DIR = BASE_DIR / 'merged' / 'data'
RESULT_VOTE_DATA_DIR = BASE_DIR / 'result_vote' / 'data'
SAVE_BILL_IDS_DATA_DIR = BASE_DIR / 'save_bill_ids' / 'data'