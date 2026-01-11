import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 基底檔案名稱 (已填充 0 值的中位數)
FILE_NAME_BASE = 'diabetes_processed_missing.csv' 

# 需要繪製盒鬚圖的欄位 (所有曾有 0 值並被填充的特徵)
cols_for_boxplot = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'BMI'
]

# 1. 載入資料
try:
    df = pd.read_csv(FILE_NAME_BASE)
    print(f"成功載入檔案：{FILE_NAME_BASE}。")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {FILE_NAME_BASE}。請確認該檔案名稱和路徑是否正確。")
    # 如果找不到檔案，您需要確認上一步是否成功產生該 CSV 檔案
    exit()

# 2. 繪製未限制（包含異常值）的盒鬚圖
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
plt.suptitle("各欄位原始盒鬚圖 (已填充 0 值，未限制異常值)", fontsize=16)

# 設置 3x3 的子圖佈局來容納 4 個圖
for i, col in enumerate(cols_for_boxplot):
    plt.subplot(3, 3, i + 1)
    # Seaborn 預設會顯示所有極端異常值 (小圓點)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col, fontsize=12)
    plt.ylabel(col)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 儲存圖表檔案
BOXPLOT_FILE_NAME = 'new_boxplots.png'
plt.savefig(BOXPLOT_FILE_NAME)
plt.close()

print(f"\n未限制的盒鬚圖已儲存至：{BOXPLOT_FILE_NAME}")