import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# 1. 讀取資料
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
target = "Outcome"

# 2. X, y 分開
X = df.drop(columns=[target])
y = df[target]

# 3. Train / Test Split (保持一樣的 random_state 以便比較)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. 建立 XGBoost 模型
# XGBoost 對參數比較敏感，這裡設定了一組針對小樣本醫療資料常用的參數
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# 5. 設定參數網格 (Grid Search)
# scale_pos_weight 是關鍵：原本資料 0:1 比例約為 2:1，所以設 2 代表加倍重視病人
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],           # XGBoost 不喜歡太深的樹，3-5 通常最好
    'learning_rate': [0.005, 0.01, 0.1], # 學習率：越低越慢但越精細
    'scale_pos_weight': [1, 2, 2.5],  # 控制對「有病」類別的重視程度
    'gamma': [0, 0.1, 0.2]            # 懲罰項，避免過擬合
}

print("正在使用 XGBoost 尋找最佳模型 (這可能會花一點時間)...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',  # 依然以 F1 為目標，追求 P 和 R 的雙高
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. 取得最佳模型
best_xgb = grid_search.best_estimator_
print(f"\n最佳參數: {grid_search.best_params_}")

# 7. 預測與評估
y_pred = best_xgb.predict(X_test)
y_pred_prob = best_xgb.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== XGBoost Result =====")
print("AUC:", round(auc, 4))
print("Accuracy:", round(acc, 4))
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))