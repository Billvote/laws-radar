# 서영소리광성fiterd_merge.csv에 컬럼구분 쉼표 외의 쉼표 제거

# 파일 경로 설정
input_file = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\billview\filter\서영소리광성fiterd_merge.csv"
output_file = r"C:\Users\1-02\Desktop\DAMF2\laws-radar\billview\filter\fiterd_merge_comma.csv"

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        # 줄 끝 개행문자 제거
        line = line.rstrip('\n')
        # 4개의 콤마까지만 split (즉, 5개 필드로 분리)
        parts = line.split(',', 4)
        if len(parts) == 5:
            # 5번째 필드(즉, content 등)에서만 콤마 제거
            parts[4] = parts[4].replace(',', '')
        # 다시 합쳐서 저장
        fout.write(','.join(parts) + '\n')
