# 설비 고장 문장 생성기

이 프로그램은 OpenAI GPT-4o API를 사용하여 설비 고장 보고서 학습 데이터를 자동 생성합니다.

## 주요 기능

- **Excel 데이터 로드**: `dictionary_data.xlsx`의 4개 시트에서 데이터 자동 읽기
- **대량 문장 생성**: 10,000개 문장 생성 지원
- **자연스러운 문체**: 한영 혼용, 띄어쓰기 오류, 오타 포함
- **안정적인 처리**: 재시도 로직 및 에러 처리 포함
- **다양한 출력**: JSONL과 CSV 형식 동시 지원

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 설정
OpenAI API 키를 환경변수로 설정:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Excel 파일 준비
`dictionary_data.xlsx` 파일에 다음 시트들이 있어야 합니다:
- 위치
- 설비유형
- 현상코드
- 우선순위

### 4. 실행
```bash
python generate_fault_sentences.py
```

## 출력 파일

- `generated_dataset.jsonl`: Fine-tuning용 JSONL 형식
- `generated_dataset.csv`: 검토용 CSV 형식

## 설정 변경

코드 내 다음 변수들을 수정하여 설정 변경 가능:
- `target_num`: 생성할 문장 수 (기본: 10,000)
- `missing_field_ratio`: 각 필드별 미입력 비율 (기본: 0.15 = 15%)
- `BATCH_SIZE`: 배치 크기 (기본: 50)

### 미입력 처리 기능
- 각 필드(위치, 설비유형, 현상코드, 우선순위)가 "미입력"으로 설정될 확률을 조정 가능
- 적어도 하나의 필드는 항상 입력된 상태로 유지
- 실제 현장에서 불완전한 입력 데이터를 반영한 학습 데이터 생성

## 예상 실행 시간

10,000개 문장 생성 시 약 30-60분 소요 (API 응답 속도에 따라 변동)