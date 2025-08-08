import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# 모델 초기화 (업로더와 동일한 모델로 유지)
model = SentenceTransformer("intfloat/multilingual-e5-base")


# 컬렉션 선택 및 검색 수행
def check_collections():
    client = QdrantClient(host="localhost", port=6333)
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]

    print("Available collections:")
    for idx, name in enumerate(collection_names):
        print(f"{idx + 1}. {name}")

    collection_choice = int(input(f"Select a collection (1-{len(collection_names)}): "))
    selected_collection = collection_names[collection_choice - 1]

    print(f"Selected collection: {selected_collection}")

    # 필드 선택
    field = input("Which field would you like to search? (설비유형/위치/현상코드): ").strip()

    # 질의 입력 및 벡터화
    query_text = input(f"Enter a search query for {field}: ").strip()
    query_vector = model.encode([query_text]).tolist()[0]

    # 검색 수행
    search_in_collection(selected_collection, query_vector, field)


# 벡터 검색 (type 필터 적용)
def search_in_collection(collection_name, query_vector, field_type):
    client = QdrantClient(host="localhost", port=6333)

    start_time = time.time()

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value=field_type)
                )
            ]
        ),
        limit=5
    )

    elapsed_time = time.time() - start_time

    print("Search results:")
    for result in search_result:
        print(f"ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")

    print(f"검색소요 시간: {elapsed_time:.4f} 초.")


# 실행
if __name__ == "__main__":
    check_collections()
