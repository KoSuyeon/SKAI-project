# Qdrant를 통한 VDB 구축 및 정규화 test
intfloat/multilingual-e5-base 모델를 사용하여 Qdranst를 구축하고 검색 성능을 테스트합니다.

### 기능 설명
- 다국어 의미 기반 임베딩: **intfloat/multilingual-e5-base** 사용
- Qdrant: 고속 벡터 검색 인덱스 관리
- Top-k 검색 평가: Top-1, Top-2 정확도 및 시간 측정
- 현상코드 특화 전처리: '설비에 ~ 현상이 발생했습니다.' 형식으로 문장화하여 검색 성능 개선
- 동의어 정규화 (선택 적용): "망가짐" → "고장" 같은 의미 동일 처리 가능

### 적재 데이터 설명
1. 입력 데이터는 이미 label 컬럼 기준으로 분류되어 있음 → 예: 설비유형, 위치, 현상코드 등의 카테고리
2. 각 row마다 다음 컬럼이 존재해야 함:
    - label: 데이터 종류 구분 (예: 설비유형)
    - input: 사용자가 입력할 것으로 예상되는 질의
    - true_name: 정규화된 최종 표준값

### VDB 적재 과정
1. input 텍스트를 문장 임베딩 모델(intfloat/multilingual-e5-base)을 사용해 벡터화
2. Qdrant 저장 시 **payload**에 아래 정보 포함:
    - "type": 해당 레이블 종류 (설비유형 / 위치 / 현상코드 등)
    - "value": 정답 정규화 텍스트 (true_name 컬럼 값)
2. 벡터는 Qdrant 컬렉션에 저장되고, 이후 검색에서 사용 가능
3. 예시 Payload:
    - {"type": "설비유형", "value": "[AGAB]Gas Detector/ Hydrocarbon (H2 포함)"}
4. 모든 데이터는 지정된 컬렉션 이름(dic_table)에 일괄 저장

### 검색 과정
1. payload 기반으로 카테고리 단위로 검색 제한
2. Top-2 검색 결과를 뽑는다.
3. Top-1과 Top-2의 유사도 차이가 거의 없으면 → 모호한 검색으로 판단
4. 이 경우 Top-2도 **정규화 후보값**으로 기록
5. Top-1이 정답이 아니고, Top-2가 정답이면 → 후보 성공으로 간주
6. 후보 포함 정확도를 별도로 계산

### 파일 설명
| 파일명                       | 설명                                     |
| ------------------------- | -------------------------------------- |
| `check_max_token.py`      | 입력 텍스트의 max token 수 측정 (모델 입력 제한 확인용)  |
| `vector_uploader.py`      | CSV 데이터를 벡터로 변환하여 Qdrant에 업로드          |
| `vector_search.py`        | 단일 질의 기반의 벡터 검색 확인용 실행 파일         |
| `vector_search_test.py`   | CSV 데이터셋을 기반으로 일괄 검색 및 정확도 평가 수행       |
| `vector_search_result.py` | 검색 결과 CSV를 분석해 label별 정확도, 평균 검색 시간 요약 |


## Label별 검색 정확도 및 시간 요약

1) 정규화 용어 원본 사용


| label | top1\_accuracy | top2\_accuracy | combined\_accuracy | avg\_search\_time | total\_count |
| ----- | -------------- | -------------- | ------------------ | ----------------- | ------------ |
| 설비유형  | 100.00%        | 0.00%          | 100.00%            | 0.0041 sec        | 163          |
| 위치    | 100.00%        | 0.00%          | 100.00%            | 0.0042 sec        | 429          |
| 현상코드  | 91.67%         | 0.00%          | 91.67%             | 0.0039 sec        | 12           |

2) 사람이 입력할 것으로 예상되는 용어 (*llm 학습 이후 test 예정)

| label | top1\_accuracy | top2\_accuracy | combined\_accuracy | avg\_search\_time | total\_count |
| ----- | -------------- | -------------- | ------------------ | ----------------- | ------------ |
| 설비유형  | 64.33%         | 7.38%          | 71.71%             | 0.0042 sec        | 813          |
| 위치    | 81.75%         | 2.50%          | 84.24%             | 0.0051 sec        | 1282         |
| 현상코드  | 0.00%          | 0.00%          | 0.00%              | 0.0044 sec        | 24           |

### 자세한 환경 구축 및 연구과정
https://www.notion.so/VDB-2460c0c062f880c982ddcae834f53b8e?source=copy_link
