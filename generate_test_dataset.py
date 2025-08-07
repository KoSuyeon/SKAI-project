import os
import json
import time
import re
import logging
import openai
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenRouter API 설정
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("환경변수 OPENROUTER_API_KEY가 설정되어 있지 않습니다.")
openai.api_key = api_key
openai.api_base = "https://openrouter.ai/api/v1"
model = "qwen/qwen-2.5-72b-instruct:free"

# 유형별 프롬프트 템플릿
def make_prompt(name, type_num, label):
    examples = {
        1: f"{name}을 모두 소문자로 바꾼 표현",
        2: f"{name}을 모두 대문자로 바꾼 표현",
        3: f"{name}에서 공백만 제거한 표현",
        4: f"{name}에서 특수기호를 제거한 표현",
        5: f"{name}에서 공백과 특수기호를 제거하고 소문자로 만든 표현",
        6: f"{name}에서 접두어(예: [AGAB])가 있다면 제거한 표현",
        7: f"{name}을 사람들이 자주 쓰는 표현으로 대체 (예: 유의어, 음역, 도메인 표현 등)"
    }
    
    return f"""
'{name}'이라는 {label} 용어를 다음 조건에 맞게 변형해줘.

조건: {examples[type_num]}

- 출력은 쉼표로 구분된 최대 5개의 표현으로 해줘.
- 줄바꿈 없이 표현만 출력해줘.
"""

# 생성형모델 호출 함수
def call_openrouter(prompt, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "너는 산업 현장에서 자주 입력되는 설비/위치/현상 용어 오류 패턴을 잘 아는 전문가야."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            #print("Raw response:", response)  # 응답 구조 확인용 출력
            content = response["choices"][0]["message"]["content"]
            variants = [v.strip() for v in content.split(",") if v.strip()]
            return variants
        except Exception as e:
            logger.warning(f"[Deepseek 오류] 재시도 {attempt+1}/{retries} - {e}")
            time.sleep(delay * (attempt + 1))
    return []

# 단일 name에 대해 type 1~7 처리
def generate_variants(name, label):
    rows = []
    for type_num in range(1, 8):  # type 1~7
        prompt = make_prompt(name, type_num, label)
        variants = call_openrouter(prompt)

        if not variants:
            logger.warning(f"[생성 결과 없음] '{name}' | label: {label}, type: {type_num}")
            continue

        for v in variants:
            if v.strip() == name.strip():
                continue  # 언어모델이 원본을 그대로 낼 경우는 무시
            rows.append({
                "input": v,
                "expected_name": name,
                "type": type_num,
                "label": label
            })

    # type 0 추가 (원본 그대로)
    rows.append({
        "input": name,
        "name": name,
        "type": 0,
        "label": label
    })
    return rows

# 전체 테스트셋 생성
def generate_dataset(names, label):
    dataset = []

    for name in tqdm(names, desc=f"{label} 처리 중"):
        dataset.extend(generate_variants(name, label))
    return pd.DataFrame(dataset)

# 엑셀에서 정답값 불러오기
def load_excel_data(file_path="dictionary_data.xlsx"):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    xls = pd.ExcelFile(file_path)
    required_sheets = ['위치', '설비유형', '현상코드', '우선순위']
    
    for sheet in required_sheets:
        if sheet not in xls.sheet_names:
            raise ValueError(f"필수 시트가 없습니다: {sheet}")
    
    df_location = pd.read_excel(xls, sheet_name='위치')
    df_equipment = pd.read_excel(xls, sheet_name='설비유형')
    df_phenomenon = pd.read_excel(xls, sheet_name='현상코드')
    df_priority = pd.read_excel(xls, sheet_name='우선순위')
    
    location_list = df_location.iloc[:, 0].dropna().astype(str).tolist()
    equipment_list = df_equipment.iloc[:, 0].dropna().astype(str).tolist()
    phenomenon_list = df_phenomenon.iloc[:, 0].dropna().astype(str).tolist()
    priority_list = df_priority.iloc[:, 0].dropna().astype(str).tolist()

    logger.info(f"로드 완료 - 위치 {len(location_list)}개 / 설비유형 {len(equipment_list)}개 / 현상코드 {len(phenomenon_list)}개 / 우선순위 {len(priority_list)}개")
    return location_list, equipment_list, phenomenon_list, priority_list


# 엑셀 파일 경로 지정 (기본은 현재 디렉토리)
location_list, equipment_list, phenomenon_list, priority_list = load_excel_data("dictionary_data.xlsx")

# 도메인별 모델 기반 variant 생성
df_equipment = generate_dataset(equipment_list, "설비유형")
df_location = generate_dataset(location_list, "위치")
df_phenomenon = generate_dataset(phenomenon_list, "현상코드")
df_priority = generate_dataset(priority_list, "우선순위")


# 병합 및 저장
df_all = pd.concat([df_equipment, df_location, df_phenomenon, df_priority], ignore_index=True)
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "normalization_testset.csv"
df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

logger.info(f"✅ 테스트셋 생성 완료 - CSV: {csv_path}")