import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 載入原始資料
df = pd.read_csv("diabetes_with_log_values.csv")  # 用完整資料才看得出差異！

# 2. 依照 Outcome 分組
df0 = df[df["Outcome"] == 0].drop(columns=["Outcome"])
df1 = df[df["Outcome"] == 1].drop(columns=["Outcome"])

# 3. 分別計算相關係數矩陣
corr0 = df0.corr(method='pearson')
corr1 = df1.corr(method='pearson')


# 4. 畫 Outcome=0 熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr0,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=.5
)
plt.title("Correlation Heatmap (Outcome = 0)")
plt.savefig("heatmap_outcome_0.png")
plt.show()


# 5. 畫 Outcome=1 熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr1,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=.5
)
plt.title("Correlation Heatmap (Outcome = 1)")
plt.savefig("heatmap_outcome_1.png")
plt.show()
