import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 載入原始數據檔案
file_name = "diabetes_log_transformed.csv"
try:
    df_original = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {file_name}。請確認檔案名稱和路徑是否正確。")
    exit()

# 2. 數據清理：將生理測量欄位中的 0 替換為 NaN
# Pima Indians Diabetes 數據集中，0 實際代表缺失值 (Missing Value)
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_cleaned = df_original.copy()
df_cleaned[cols_to_replace] = df_cleaned[cols_to_replace].replace(0, np.nan)

# 顯示清理後的缺失值摘要
print("--- 清理後的缺失值摘要 ---")
print(df_cleaned.isnull().sum())
print("--------------------------")

# 3. 計算相關係數矩陣 (Pandas會自動使用成對刪除處理NaN)
# method='pearson' 是計算線性關係的標準方法
correlation_matrix = df_cleaned.corr(method='pearson')

# 4. 繪製相關性熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, # 顯示相關係數數值
    fmt=".2f",  # 數值保留兩位小數
    cmap='coolwarm', # 使用藍紅色系熱圖
    linewidths=.5, # 增加格子線
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)

# 設置標題，明確指出處理方法
plt.title('Correlation Heatmap (Using cleaned Data with Pairwise Deletion)')
plt.show() # 在 Notebook 環境中顯示圖像
# 如果需要儲存圖像，請取消註解下面一行
# plt.savefig('correlation_original_heatmap.png')

# 5. 輸出 Insulin 欄位的關鍵相關性數據
print("\n--- Insulin 欄位的關鍵相關性 (與 Outcome 和 Glucose) ---")
print(correlation_matrix[['Insulin', 'Outcome', 'Glucose']].loc['Insulin'].drop('Insulin'))
print("----------------------------------------------------------")