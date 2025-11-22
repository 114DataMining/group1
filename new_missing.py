import pandas as pd
import numpy as np
# 讀取資料 (假設檔案名為 diabetes.csv)
df = pd.read_csv('diabetes.csv')

# 步驟 1: 將指定欄位中的 0 視為 NaN (缺失值)
# 注意：Insulin 也在這個列表中，雖然下一步會刪除，但照邏輯先標記為 NaN 也可以
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# 步驟 2: 刪除 Insulin 欄位 (因為缺值過多)
df.drop(columns=['Insulin'], inplace=True)

# 步驟 3: 使用中位數填補 Glucose, BloodPressure, BMI
# 注意：SkinThickness 不在此列，因為您計畫之後用插值/KNN處理
cols_to_fill_median = ['Glucose', 'BloodPressure', 'BMI']

for col in cols_to_fill_median:
    # 計算該欄位的中位數 (會自動忽略 NaN)
    median_val = df[col].median()
    # 填補缺失值
    df[col].fillna(median_val, inplace=True)

# 檢查結果：查看這三個欄位是否還有 NaN
print(df[cols_to_fill_median].isnull().sum())
df.to_csv('diabetes_processed_missing.csv', index=False)