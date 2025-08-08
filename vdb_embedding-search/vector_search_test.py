import pandas as pd
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# 설정
MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "dic_table"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
#CSV_PATH = "data/vdb_search_test_raw.csv"
#SAVE_PATH = "data/vdb_search_test_raw_results.csv
CSV_PATH = "data/vdb_search_test.csv"
SAVE_PATH = "data/vdb_search_test_results.csv"
SCORE_DIFF_THRESHOLD = 0.01  # 후보로 인정할 score 차이

# 초기화
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

df = pd.read_csv(CSV_PATH)
results = []

def enrich_input(text):
    return f"설비에 '{text}' 현상이 발생했습니다."

for idx, row in df.iterrows():

    input_text = row["input"].strip()
    field = row["label"].strip()
    true_name = str(row["true_name"]).strip()

    # 현상코드일 때만 enrich
    if field == "현상코드":
        query_text = enrich_input(input_text)
    else:
        query_text = input_text

        
    query_vector = model.encode([query_text])[0]

    # 검색 (Top-2)
    start = time.time()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[FieldCondition(
                key="type",
                match=MatchValue(value=field)
            )]
        ),
        limit=2,
        with_payload=True,
        with_vectors=False
    )
    elapsed = time.time() - start

    # 결과 처리
    top1_value, top2_value = None, None
    top1_score, top2_score = None, None
    is_correct_top1, is_correct_top2 = False, False

    if len(search_result) > 0:
        top1 = search_result[0]
        top1_value = top1.payload.get("value", "").strip()
        top1_score = top1.score
        is_correct_top1 = (top1_value == true_name)

    if len(search_result) > 1:
        top2 = search_result[1]
        top2_value = top2.payload.get("value", "").strip()
        top2_score = top2.score

        # 후보로 인정할 정도로 유사한 경우만 평가
        if abs(top1_score - top2_score) < SCORE_DIFF_THRESHOLD:
            is_correct_top2 = (top2_value == true_name)

    # 결과 저장
    results.append({
        "label": field,
        "input": query_text,
        "true_name": true_name,
        "top1_value": top1_value,
        "top1_score": top1_score,
        "is_correct_top1": is_correct_top1,
        "top2_value": top2_value,
        "top2_score": top2_score,
        "is_correct_top2": is_correct_top2,
        "search_time_sec": round(elapsed, 4)
    })

# 저장
df_result = pd.DataFrame(results)
df_result.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 결과 저장 완료: {SAVE_PATH}")

# 정확도 출력
top1_acc = df_result["is_correct_top1"].mean()
combined_acc = ((df_result["is_correct_top1"]) | (df_result["is_correct_top2"])).mean()
avg_time = df_result["search_time_sec"].mean()

print(f"\n📊 평가 결과:")
print(f"🔹 Top-1 정확도: {top1_acc:.2%}")
print(f"🔹 Top-2 포함 정확도: {combined_acc:.2%}")
print(f"⏱️ 평균 검색 시간: {avg_time:.4f}초")