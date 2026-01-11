import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt  # 引入繪圖套件
import numpy as np # 新增 numpy 用於數學計算
import joblib # 用於存檔
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
    recall_score,
    precision_score,
    f1_score,
    ConfusionMatrixDisplay # 新增：用於畫漂亮矩陣
)

# 1. 讀取資料
# 請確認檔案路徑是否正確
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

# 4. 建立 XGBoost 模型
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# 5. 設定參數網格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [1, 2, 3],  # 針對醫療資料，通常嘗試調高此值
    'gamma': [0, 0.1, 0.2]
}

print("正在使用 XGBoost 尋找最佳模型 (GridSearchCV)...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',  # 這裡改用 AUC 作為搜尋標準，通常醫療專案更看重 AUC
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. 取得最佳模型
best_xgb = grid_search.best_estimator_
print(f"\n最佳參數: {grid_search.best_params_}")

# 7. 預測與評估
# 先取得預測為 1 (有病) 的機率
y_pred_prob = best_xgb.predict_proba(X_test)[:, 1] 

# --- 【關鍵修改】設定自訂閥值 ---
THRESHOLD = 0.7
# 如果機率 >= 0.7 則判定為 1 (有病)，否則為 0 (健康)
y_pred_custom = (y_pred_prob >= THRESHOLD).astype(int)

print(f"\n目前設定的閥值 (Threshold): {THRESHOLD}")

# --- 計算各項指標 (使用 y_pred_custom) ---
acc = accuracy_score(y_test, y_pred_custom)
# AUC 是看機率分佈的，跟閥值無關，所以不用重算，或是維持原樣
roc_auc = roc_auc_score(y_test, y_pred_prob) 
precision = precision_score(y_test, y_pred_custom, zero_division=0)
recall = recall_score(y_test, y_pred_custom) # Sensitivity
f1 = f1_score(y_test, y_pred_custom)

# 計算 Specificity (特異度)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom).ravel()
specificity = tn / (tn + fp)

print(f"\n===== XGBoost 詳細評估報告 (Threshold = {THRESHOLD}) =====")
print(f"AUC (Area Under Curve):  {roc_auc:.4f}  (AUC 與閥值無關，數值不變)")
print(f"Accuracy (準確率):       {acc:.4f}")
print(f"Precision (精確率):      {precision:.4f}  (門檻提高，這裡通常會變高)")
print(f"Recall (召回率/敏感度):   {recall:.4f}  (門檻提高，這裡通常會變低)")
print(f"Specificity (特異度):    {specificity:.4f}  (門檻提高，這裡通常會變高)")
print(f"F1-Score:               {f1:.4f}")
print("\nConfusion Matrix (混淆矩陣):")
print(confusion_matrix(y_test, y_pred_custom))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom, zero_division=0))

# --- 8. 繪製 ROC 曲線 ---
print("\n(1/4) 正在繪製 ROC 曲線...")
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_xgb, X_test, y_test)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title(f'ROC Curve - XGBoost Model (Threshold: {THRESHOLD})')
plt.grid(True)
plt.show() # 記得關掉視窗才會跑下一張圖

# ==========================================
# 9. (新增) 繪製學習曲線 (Learning Curve)
# ==========================================
print("\n(2/4) 正在繪製學習曲線 (Learning Curve)...")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (AUC)")
    
    # 呼叫 sklearn 的 learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    
    # 計算平均值與標準差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    # 畫出區域
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # 畫線
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

plot_learning_curve(best_xgb, "Learning Curve (XGBoost)", X_train, y_train, cv=5)
plt.show() # 記得關掉視窗才會跑下一張圖

# ==========================================
# 10. (新增) 繪製彩色混淆矩陣
# ==========================================
print("\n(3/4) 正在繪製混淆矩陣圖 (Confusion Matrix)...")
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (0)", "Diabetic (1)"])
disp.plot(cmap=plt.cm.Blues, values_format='d') 
plt.title(f'Confusion Matrix (Threshold = {THRESHOLD})')
plt.show() # 記得關掉視窗才會跑下一張圖

# ==========================================
# 11. 過擬合檢查 (Overfitting Check)
# ==========================================
print("\n===== 過擬合檢查 (Overfitting Check) =====")
# 對訓練集做預測
y_train_pred_prob = best_xgb.predict_proba(X_train)[:, 1]
y_train_pred_custom = (y_train_pred_prob >= THRESHOLD).astype(int)

# 計算訓練集的 AUC 與 Accuracy
train_auc = roc_auc_score(y_train, y_train_pred_prob)
train_acc = accuracy_score(y_train, y_train_pred_custom)

# 顯示對比
print(f"訓練集 AUC: {train_auc:.4f}  vs  測試集 AUC: {roc_auc:.4f}")
print(f"訓練集 Acc: {train_acc:.4f}  vs  測試集 Acc: {acc:.4f}")

# 自動判斷
diff = train_auc - roc_auc
if diff > 0.1:
    print(f"⚠️ 警告：差距 {diff:.4f} > 0.1，模型可能有過擬合 (Overfitting) 現象！")
else:
    print(f"✅ 狀態良好：差距 {diff:.4f}，模型學習狀況健康。")


# ==========================================
# 12. 繪製特徵重要性
# ==========================================
print("\n(4/4) 正在繪製特徵重要性 (Feature Importance)...")

# 設定圖片大小
plt.figure(figsize=(10, 8))

# 繪製特徵重要性圖表 (使用 'gain' 看貢獻度)
# max_num_features=10 表示只顯示前 10 個最重要的特徵
xgb.plot_importance(best_xgb, max_num_features=10, importance_type='gain',
                    title='Top 10 Feature Importance (Gain)', xlabel='Gain Value')

plt.show()

# 儲存模型
joblib.dump(best_xgb, 'my_diabetes_xgb_model_0.7.pkl')
print("\n模型已儲存為 'my_diabetes_xgb_model_0.7.pkl'")