import pandas as pd

# 讀取資料
df = pd.read_csv("diabetes_processed_missing.csv")

# 建立一份副本來移除離群值
df_no_outliers = df.copy()

# 要分析的欄位
columns = ["Glucose", "BloodPressure", "BMI", "SkinThickness"]

# 建立總表用的空清單
all_outliers = []

# 逐一分析欄位
for column in columns:
    print(f"\n=== 欄位：{column} ===")

    # 計算IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # 找出離群值
    outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]

    # 印出分析結果
    print(f"離群值數量：{len(outliers)}")

    if len(outliers) > 0:
        index_list = outliers.index.tolist()
        value_list = outliers[column].tolist()

        print("離群值序號：", ", ".join(map(str, index_list)))
        print("離群值數值：", ", ".join(map(lambda x: f"{x:.2f}", value_list)))

        # 輸出該欄位的離群值至 CSV
        outliers_to_save = outliers[[column]].copy()
        outliers_to_save["Index"] = outliers_to_save.index
        outliers_to_save["Column"] = column
        outliers_to_save["Lower Limit"] = lower_limit
        outliers_to_save["Upper Limit"] = upper_limit

        outliers_to_save.to_csv(f"outliers_{column}.csv", index=False, encoding="utf-8-sig")
        print(f"→ 已將 {column} 的離群值輸出到 outliers_{column}.csv")

        # 加入總表
        all_outliers.append(outliers_to_save)

        # 🚨**從資料中刪除這些離群值**
        df_no_outliers = df_no_outliers[~df_no_outliers.index.isin(index_list)]
    else:
        print("沒有離群值。")

# 若有任何離群值則合併輸出總表
if all_outliers:
    all_outliers_df = pd.concat(all_outliers).reset_index(drop=True)
    all_outliers_df.rename(columns={columns[0]: "Outlier Value"}, inplace=False)
    all_outliers_df.to_csv("all_outliers_summary.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 所有欄位離群值已輸出至：all_outliers_summary.csv")
else:
    print("\n🎉 所有欄位都沒有離群值。")

# 🚨　**輸出刪除離群值後的新資料**
df_no_outliers.to_csv("diabetes_no_outliers.csv", index=False, encoding="utf-8-sig")
print("\n✅ 已輸出刪除離群值後的新檔案：diabetes_no_outliers.csv")
