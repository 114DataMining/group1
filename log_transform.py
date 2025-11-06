import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取資料
df = pd.read_csv("diabetes_cleaned_before_after_zeros.csv")

# 目標欄位
columns = ["Insulin", "SkinThickness"]

# 建立一個新 DataFrame 來儲存轉換後結果
df_log = df.copy()

for col in columns:
    non_zero_min = df[df[col] > 0][col].min()
    df_log[col] = np.where(df[col] <= 0, non_zero_min / 2, df[col])
    df_log[col] = np.log(df_log[col])

# ✅ 儲存 Log 轉換後的資料
output_csv = "diabetes_log_transformed.csv"
df_log.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"✅ 已將 Log 轉換後的資料儲存為：{output_csv}")

# 繪圖比較前後分布
plt.figure(figsize=(10, 6))
for i, col in enumerate(columns):
    plt.subplot(2, 2, i*2 + 1)
    sns.histplot(df[col], kde=True, color="skyblue")
    plt.title(f"{col} - 原始分布")

    plt.subplot(2, 2, i*2 + 2)
    sns.histplot(df_log[col], kde=True, color="orange")
    plt.title(f"{col} - Log轉換後分布")

# ✅ 儲存圖檔
plt.tight_layout()
plt.savefig("log_transform_result.png")
plt.close()
print("📊 圖片已儲存為：log_transform_result.png")
