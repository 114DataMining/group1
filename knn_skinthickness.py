import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# 1. 讀取你已經做過 Min-Max 的資料
df = pd.read_csv("diabetes_minmax_normalized.csv")

# 2. SkinThickness 可能原本用 0 表示缺失 → 改成 NaN
df["SkinThickness"] = df["SkinThickness"].replace(0, pd.NA)
df = df.replace({pd.NA: np.nan})
# 3. 建立 imputer，要補哪個欄位？
target = "SkinThickness"

# KNN 要的特徵（用你之前說的：血糖、血壓、BMI、年齡、血緣功能）
features = ["Glucose", "BloodPressure", "BMI", "Age", "DiabetesPedigreeFunction", target]

df_subset = df[features].copy()

df = df.replace({pd.NA: np.nan})
# 或簡寫
# df = df.astype('float64')

# 4. 建立 KNN 補值器
imputer = KNNImputer(n_neighbors=5)

# 5. 執行補值（注意：這裡的 SkinThickness 是 0~1 標準化後的值）
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_subset),
    columns=features
)

# 6. 取出補好的 SkinThickness（仍是 0~1）
df[target] = df_imputed[target]

# 7.（重要）把 SkinThickness 反標準化
# 取得原始 SkinThickness 的 min/max
orig_df = pd.read_csv("diabetes_no_outliers.csv")  # 用補值前的原始檔
orig_skin_min = orig_df["SkinThickness"].min()
orig_skin_max = orig_df["SkinThickness"].max()

df[target] = df[target] * (orig_skin_max - orig_skin_min) + orig_skin_min

# 8. 存成新的 csv
df.to_csv("diabetes_skinthickness_knn_imputed.csv", index=False, encoding="utf-8-sig")

print("✔ 補值完成！已輸出 diabetes_skinthickness_knn_imputed.csv")
