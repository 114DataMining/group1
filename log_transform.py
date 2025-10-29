# log_transform_insulin_skin.py
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
    # 為避免 log(0) 錯誤，將 0 或負值轉為 NaN，再補成最小非零值的一半
    non_zero_min = df[df[col] > 0][col].min()
    df_log[col] = np.where(df[col] <= 0, non_zero_min / 2, df[col])
    df_log[col] = np.log(df_log[col])

# 繪圖比較前後分布
plt.figure(figsize=(10, 6))
for i, col in enumerate(columns):
    plt.subplot(2, 2, i*2 + 1)
    sns.histplot(df[col], kde=True, color="skyblue")
    plt.title(f"{col} - 原始分布")

    plt.subplot(2, 2, i*2 + 2)
    sns.histplot(df_log[col], kde=True, color="orange")
    plt.title(f"{col} - Log轉換後分布")

plt.tight_layout()
plt.show()
