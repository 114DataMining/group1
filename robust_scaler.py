# robust_scaler_insulin_skin.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# 讀取資料
df = pd.read_csv("diabetes_cleaned_before_after_zeros.csv")

# 目標欄位
columns = ["Insulin", "SkinThickness"]

# 使用 RobustScaler
scaler = RobustScaler()
df_scaled = df.copy()
df_scaled[columns] = scaler.fit_transform(df[columns])

# 繪圖比較前後分布
plt.figure(figsize=(10, 6))
for i, col in enumerate(columns):
    plt.subplot(2, 2, i*2 + 1)
    sns.histplot(df[col], kde=True, color="lightgreen")
    plt.title(f"{col} - 原始分布")

    plt.subplot(2, 2, i*2 + 2)
    sns.histplot(df_scaled[col], kde=True, color="tomato")
    plt.title(f"{col} - RobustScaler後分布")

plt.tight_layout()
plt.show()
