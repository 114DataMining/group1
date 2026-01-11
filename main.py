import pandas as pd
import numpy as np

# 載入資料集
df = pd.read_csv('diabetes.csv')

# 定義需要將 0 視為缺失值的欄位
cols_to_impute = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

# --- 步驟 0: 顯示清理前 0 值的數量 ---
print("\n--- 清理前 0 值的數量 (這些將被視為缺失值) ---")
initial_zeros = df[cols_to_impute].eq(0).sum()
print(initial_zeros)

# --- 步驟 1: 計算中位數 (排除 0 值) ---
# 創建一個臨時 DataFrame，將 0 替換為 NaN，以便計算正確的中位數
df_temp = df.copy()
df_temp[cols_to_impute] = df_temp[cols_to_impute].replace(0, np.nan)

medians = {}
print("\n--- 需要用來填充缺失值 (0 值) 的各欄位中位數 ---")
for col in cols_to_impute:
    median_val = df_temp[col].median()
    medians[col] = median_val
    print(f"欄位 '{col}': 中位數 = {median_val}")


# --- 步驟 2: 執行資料清理 (替換 0 為中位數) ---
# 將原始 df 中的 0 值替換為 NaN

df[cols_to_impute] = df[cols_to_impute].replace(0, np.nan)

# 使用計算出的中位數填充缺失值 (NaN)
for col in cols_to_impute:
    df[col].fillna(medians[col], inplace=True)

# --- 步驟 3: 顯示清理後 0 值的數量 ---
print("\n--- 清理後 0 值的數量 (驗證成功，應為 0) ---")
final_zeros = df[cols_to_impute].eq(0).sum()
print(final_zeros)

# 將清理後的資料儲存為新的 CSV 檔案
df.to_csv('diabetes_cleaned_before_after_zeros111.csv', index=False)