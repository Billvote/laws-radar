import pandas as pd
import sys
from pathlib import Path
# settings.py를 불러오기 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
import settings
import datetime

card_path = settings.BASE_DIR / 'geovote' / 'data' / 'card_news_output.csv'
bill_path = settings.BASE_DIR / 'geovote' / 'data' / 'urls.csv'


bill_df = pd.read_csv(bill_path)

card_df = pd.read_csv(card_path)
card_df = card_df[['bill_id', 'card_news_content']]

# print(card_df.head())
# print(bill_df.head())

merged_df = pd.merge(bill_df, card_df, how='inner', on='bill_id')
# print(merged_df.head())
# print(merged_df.columns)

csv_path = settings.BASE_DIR / 'geovote' / 'data' / 'card_news.csv'
merged_df.to_csv(csv_path, index=False, na_rep='NULL')