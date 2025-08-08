import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# 1. 설정
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "dic_table"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base" 

# 2. 벡터 생성 함수
def create_embedding(texts, model):
    return model.encode(texts).tolist()

# 3. Qdrant 초기화
def init_qdrant():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    return client

# 4. CSV 또는 Excel 로드 및 전처리
def load_table(filepath):
    # 시트별로 읽기
    df_equipment = pd.read_excel(filepath, sheet_name=0)  # 첫 번째 시트
    df_location = pd.read_excel(filepath, sheet_name=1)  # 두 번째 시트
    df_state = pd.read_excel(filepath, sheet_name=2)  # 세 번째 시트
    return df_equipment, df_location, df_state

import uuid

# 5. Point 생성 (각 필드 별로 벡터화 처리)
def generate_points(df_equipment, df_location, df_state, model):
    equipment_texts = df_equipment["설비유형"].fillna("").astype(str).tolist()
    location_texts = df_location["위치"].fillna("").astype(str).tolist()
    state_texts = df_state["현상코드"].fillna("").astype(str).tolist()

    equipment_vectors = create_embedding(equipment_texts, model)
    location_vectors = create_embedding(location_texts, model)
    state_vectors = create_embedding(state_texts, model)

    max_len = max(len(df_equipment), len(df_location), len(df_state))

    points = []
    for idx in range(max_len):
        if idx < len(equipment_vectors) and equipment_vectors[idx]:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=equipment_vectors[idx],
                payload={
                    "type": "설비유형",
                    "value": df_equipment.iloc[idx]["설비유형"]
                }
            ))

        if idx < len(location_vectors) and location_vectors[idx]:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=location_vectors[idx],
                payload={
                    "type": "위치",
                    "value": df_location.iloc[idx]["위치"]
                }
            ))

        if idx < len(state_vectors) and state_vectors[idx]:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=state_vectors[idx],
                payload={
                    "type": "현상코드",
                    "value": df_state.iloc[idx]["현상코드"]
                }
            ))

    return points


# 6. 실행
def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = init_qdrant()
    
    df_equipment, df_location, df_state = load_table(filepath="/home/syko/sk-ai-project/vdb_embedding-search/data/dictionary_data.xlsx") # 경로 설정

    points = generate_points(df_equipment, df_location, df_state, model)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"{len(points)} vectors inserted into {COLLECTION_NAME}.")

if __name__ == "__main__":
    main()
