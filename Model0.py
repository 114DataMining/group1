import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report


# 1. 讀取資料
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
target = "Outcome"


# 2. X, y 分開
X = df.drop(columns=[target])
y = df[target]


# 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. 設定參數網格 (這是給 AI 的實驗範圍)
# 這裡包含了我們討論過的策略：加多樹、調整葉子大小、調整權重
param_grid = {
    'n_estimators': [50, 100, 200],       # 嘗試種更多樹來穩定結果
    'max_depth': [10, 15, None],           # 讓樹有機會長得更深一點
    'min_samples_leaf': [1, 2, 4],         # 關鍵：1 會提升 Recall, 4 會提升 Precision
    'class_weight': ['balanced', {0:1, 1:2}] # 比較「系統自動平衡」與「手動加重病人權重(2倍)」哪個好
}


# 5. 建立基礎模型
rf = RandomForestClassifier(random_state=42)


# 6. 啟動 GridSearchCV (超級實驗助手)
# scoring='f1' 是核心：告訴電腦我們要找的是「綜合表現(F1)」最好的組合
print("正在尋找最佳參數，請稍候...")
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,            # 5-Fold 交叉驗證，確保結果不是運氣好
    scoring='f1',    # <--- 目標：最大化 F1-Score (Precision 與 Recall 的平衡點)
    n_jobs=-1,       # 全力運轉 CPU
    verbose=1
)


grid_search.fit(X_train, y_train)


# 7. 取得冠軍模型與參數
best_rf = grid_search.best_estimator_
print(f"\n最佳參數組合: {grid_search.best_params_}")
print(f"最佳驗證 F1 分數: {grid_search.best_score_:.4f}")


# 8. 用冠軍模型進行預測
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]


# 9. 最終評估
auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("\n===== Optimized Random Forest Result =====")
print("AUC:", round(auc, 4))
print("Accuracy:", round(acc, 4))
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
