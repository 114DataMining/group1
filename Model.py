import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve

# 1. 讀取資料
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")

# 確認 target 欄位名稱（你資料中是 Outcome）
target = "Outcome"

# 2. X, y 分開
X = df.drop(columns=[target])
y = df[target]

# 3. Train / Test Split（80/20 + 分層抽樣）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 資料是分類問題 → 必須分層抽樣
)

# 4. 建立 Random Forest（預設參數 = Baseline）
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 5. 預測
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]  # ← 得病機率

# 6. 評估
auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("===== Random Forest Baseline =====")
print("AUC:", round(auc, 4))
print("Accuracy:", round(acc, 4))
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))