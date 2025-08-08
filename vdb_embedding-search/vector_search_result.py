import pandas as pd
from tabulate import tabulate

try:
    # CSV 경로 확인
    df_result = pd.read_csv("data/vdb_search_test_results.csv")
    #df_result = pd.read_csv("data/vdb_search_test_raw_results.csv")


    if df_result.empty:
        print("⚠️ 데이터가 비어 있습니다. 파일을 확인해주세요.")
    else:
        # Top1 또는 Top2 중 하나라도 정답
        df_result["is_correct_combined"] = df_result["is_correct_top1"] | df_result["is_correct_top2"]

        # label별 요약
        summary = df_result.groupby("label").agg(
            top1_accuracy=("is_correct_top1", "mean"),
            top2_accuracy=("is_correct_top2", "mean"),
            combined_accuracy=("is_correct_combined", "mean"),
            avg_search_time=("search_time_sec", "mean"),
            total_count=("label", "count")
        )

        # 포맷팅
        summary["top1_accuracy"] = (summary["top1_accuracy"] * 100).round(2)
        summary["top2_accuracy"] = (summary["top2_accuracy"] * 100).round(2)
        summary["combined_accuracy"] = (summary["combined_accuracy"] * 100).round(2)
        summary["avg_search_time"] = summary["avg_search_time"].round(4)

        print("📊 Label별 검색 정확도 및 시간 요약:\n")
        print(tabulate(summary, headers="keys", tablefmt="github"))

except FileNotFoundError:
    print("❌ 'data/vdb_search_test_results.csv' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ 에러 발생: {e}")
