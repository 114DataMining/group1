import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 讀取你的資料
df = pd.read_csv("diabetes_no_outliers.csv")

# 2. 建立 MinMaxScaler（將數值壓到 0–1）
scaler = MinMaxScaler()

# 3. 只對「數值欄位」做標準化
exclude_cols = ['Pregnancies', 'Outcome']
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.difference(exclude_cols)

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. 查看結果
print(df.head())

# 5.（可選）存成新檔案
df.to_csv("diabetes_minmax_normalized.csv", index=False)
