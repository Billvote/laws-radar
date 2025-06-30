import pandas as pd

# df = pd.read_csv('bill.csv')
# df_copy = df[['cluster', 'cluster_keyword']]

# df_copy.to_csv('cluster_keywords.csv', index=False)

# df = pd.read_csv('cluster_keywords.csv')
# df_unique = df.drop_duplicates()
# # print(df_unique.head())
# # print(df_unique.shape)

# counts = df['cluster_keyword'].value_counts()
# filtered = counts[counts >= 5]

# print(filtered)

import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수 가져오기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def generate_nickname(cluster_keywords):
    # client = genai()


    model_name = "models/chat-bison-001"

    prompt_text = (
        f"이 키워드들을 바탕으로 한국어로 멋지고 의미 있는 별명을 1개 추천해줘. "
        f"키워드: {cluster_keywords}\n"
        f"별명:"
    )

    response = genai.generate_text(
        model=model_name,
        prompt=prompt_text,
        temperature=0.7,
        max_tokens=20
    )

    nickname = response.candidates[0].output.strip()
    return nickname


if __name__ == "__main__":
    example_keywords = "의료, 수급, 인력, 위원회"
    nickname = generate_nickname(example_keywords)
    print("추천 닉네임:", nickname)
