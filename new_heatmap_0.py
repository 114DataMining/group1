import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 載入完整資料（包含 outcome=0 和 1）
df = pd.read_csv("diabetes_with_log_values.csv")

# 2. 分組
df0 = df[df["Outcome"] == 0].drop(columns=["Outcome"])
df1 = df[df["Outcome"] == 1].drop(columns=["Outcome"])

# 3. 計算相關係數矩陣
corr0 = df0.corr(method='pearson')
corr1 = df1.corr(method='pearson')

# 4. Outcome=0 熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr0, annot=True, fmt=".2f",
    cmap="coolwarm", linewidths=.5
)
plt.title("Correlation Heatmap (Outcome = 0)")
plt.tight_layout()
plt.savefig("heatmap_outcome_0.png")
plt.clos
