import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # 用來顯示檔案路徑
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# 引入 Pipeline 確保順序正確
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE

# ==========================================
# 1. 讀取資料與基礎設定
# ==========================================
# 請確認 csv 檔案跟程式碼在同一個資料夾
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
target = "Outcome"

X = df.drop(columns=[target])
y = df[target]

# 2. 切分 80% 訓練集, 20% 測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("-" * 60)
print("[1] 原始資料切分狀況")
print("-" * 60)
print(f"完整訓練集 (80%): {len(y_train)} 筆")
print(f"   類別 0 (沒病): {sum(y_train==0)} 筆")
print(f"   類別 1 (有病): {sum(y_train==1)} 筆")
print(f"測試集 (20%): {len(y_test)} 筆")

# ==========================================
# 3. GridSearchCV 尋找最佳超參數
# ==========================================
print("\n" + "-" * 60)
print("[2] 執行 GridSearchCV 尋找最佳超參數")
print("-" * 60)

# 建立基礎 Pipeline
pipeline_grid = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', BorderlineSMOTE(random_state=42, kind='borderline-1')),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=None)) 
])

# 設定擴充後的超參數範圍
param_grid = {
    'rf__n_estimators': [100, 200, 300],        
    'rf__max_depth': [5, 10, None],             
    'rf__min_samples_split': [2, 5],            
    'rf__min_samples_leaf': [1, 2, 4],          
    'rf__max_features': ['sqrt', 'log2']        
}

# 執行搜尋
grid_search = GridSearchCV(
    estimator=pipeline_grid,
    param_grid=param_grid,
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 列出所有超參數配對結果
print("\n[詳細報告] 每個超參數組合的測試結果 (按 roc_auc 分數排序):")
results_df = pd.DataFrame(grid_search.cv_results_)
cols_to_show = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
print(results_df[cols_to_show].sort_values(by='rank_test_score').head(5).to_string(index=False))

# 取得最佳參數與模型
best_params = grid_search.best_params_
best_model_pipeline = grid_search.best_estimator_

print(f"\n最終選定最佳參數: {best_params}")
print(f"最佳 CV Score (roc_auc): {grid_search.best_score_:.4f}")


# ==========================================
# 4. 使用「最佳參數」執行詳細的 5-Fold CV
# ==========================================
cv_acc, cv_f1, cv_auc = [], [], []
tprs = []
mean_fpr = np.linspace(0, 1, 100)
# 第一張圖：訓練集交叉驗證 ROC
fig1, ax1 = plt.subplots(figsize=(10, 8)) 

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "-" * 60)
print("[3] 針對「最佳參數」執行 5-Fold CV 詳細分析")
print("-" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # 訓練模型
    best_model_pipeline.fit(X_tr, y_tr)
    
    # 預測
    y_pred = best_model_pipeline.predict(X_val)
    y_prob = best_model_pipeline.predict_proba(X_val)[:, 1]

    # 計算指標
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc_val = roc_auc_score(y_val, y_prob)

    cv_acc.append(acc)
    cv_f1.append(f1)
    cv_auc.append(roc_auc_val)

    # 畫 ROC 線
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    ax1.plot(fpr, tpr, alpha=0.3, label=f'ROC Fold {fold} (AUC = {roc_auc_val:.2f})')
    
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

# 繪製 CV 平均 ROC 曲線
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(cv_auc)

ax1.plot(mean_fpr, mean_tpr, color='b',
        label=f'Mean ROC (AUC = {mean_auc:.2f} +/- {std_auc:.2f})',
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label='+/- 1 std. dev.')

ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f"ROC Curve (5-Fold CV)",
        xlabel='False Positive Rate', ylabel='True Positive Rate')
ax1.legend(loc="lower right")
plt.show() 

# 印出訓練集五折平均統計
print("-" * 60)
print("[訓練集 5-Fold 平均結果]")
print(f"平均 Accuracy: {np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f}")
print(f"平均 AUC:      {np.mean(cv_auc):.4f} +/- {np.std(cv_auc):.4f}")
print(f"平均 F1 Score: {np.mean(cv_f1):.4f}  +/- {np.std(cv_f1):.4f}")


# ==========================================
# 5. 最終模型訓練與評估 (包含訓練集與測試集)
# ==========================================
print("\n" + "-" * 60)
print("[4] 最終模型訓練與成效評估")
print("-" * 60)

# 1. 用完整的 80% 訓練集重新訓練一次最佳模型
best_model_pipeline.fit(X_train, y_train)

# -------------------------------------------------------
# [A] 訓練集 (Training Set) 自我評估結果
# -------------------------------------------------------
print("\n>>> [A] 訓練集 (Training Set) 自我評估結果:")
y_train_pred = best_model_pipeline.predict(X_train)
y_train_prob = best_model_pipeline.predict_proba(X_train)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_prob)
cm_train = confusion_matrix(y_train, y_train_pred)
tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()

print(f"Accuracy: {train_acc:.4f}")
print(f"AUC:      {train_auc:.4f}")
print(f"F1 Score: {train_f1:.4f}")
print(f"混淆矩陣:\n TN={tn_tr} | FP={fp_tr}\n FN={fn_tr} | TP={tp_tr}")
print("-" * 30)


# -------------------------------------------------------
# [B] 標準測試集 (Test Set) 評估 (Threshold = 0.5)
# -------------------------------------------------------
print("\n>>> [B] 測試集 (Test Set) 結果 (Threshold = 0.5):")

y_prob_test = best_model_pipeline.predict_proba(X_test)[:, 1]
y_pred_std = best_model_pipeline.predict(X_test)

acc_std = accuracy_score(y_test, y_pred_std)
f1_std = f1_score(y_test, y_pred_std)
auc_std = roc_auc_score(y_test, y_prob_test)

print(f"Accuracy: {acc_std:.4f}")
print(f"AUC:      {auc_std:.4f}")
print(f"F1 Score: {f1_std:.4f}")

cm_std = confusion_matrix(y_test, y_pred_std)
tn, fp, fn, tp = cm_std.ravel()
print(f"混淆矩陣:\n TN={tn} | FP={fp}\n FN={fn} | TP={tp}")

# 繪製標準 ROC 曲線 (測試集)
fig2, ax2 = plt.subplots(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'Test ROC (AUC = {auc_std:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
idx_05 = np.argmin(np.abs(thresholds - 0.5))
ax2.scatter(fpr[idx_05], tpr[idx_05], s=100, c='black', label='Threshold = 0.5')

ax2.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
        title='ROC Curve (Test Set - Standard)',
        xlabel='False Positive Rate', ylabel='True Positive Rate')
ax2.legend(loc="lower right")
plt.show()

# ==========================================
# 6. 匯出機率 CSV 與 閾值調整測試 (Threshold = 0.75)
# ==========================================
print("\n" + "-" * 60)
print("[5] 匯出 CSV 與 閾值調整分析 (Threshold = 0.75)")
print("-" * 60)

# 1. 製作 DataFrame 並存檔
df_result = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted_Probability': y_prob_test,
    'Pred_0.5': y_pred_std
})
csv_name = "diabetes_test_probabilities.csv"
df_result.to_csv(csv_name, index=False)

print(f"✅ CSV 檔案已成功建立: {csv_name}")

# ==========================================
# ★ 重點修改：提高閾值到 0.75
# ==========================================
NEW_THRESHOLD = 0.75  # 從 0.65 改為 0.75，為了極致降低 FP

y_pred_new = (y_prob_test >= NEW_THRESHOLD).astype(int)

# 3. 計算新指標
acc_new = accuracy_score(y_test, y_pred_new)
f1_new = f1_score(y_test, y_pred_new)
auc_new = roc_auc_score(y_test, y_prob_test) 

print(f"\n[調整後測試集結果 (Threshold = {NEW_THRESHOLD})]")
print(f"Accuracy: {acc_new:.4f}")
print(f"AUC:      {auc_new:.4f}")
print(f"F1 Score: {f1_new:.4f}")

# 4. 新混淆矩陣
cm_new = confusion_matrix(y_test, y_pred_new)
tn_n, fp_n, fn_n, tp_n = cm_new.ravel()
print(f"新混淆矩陣 (目標：FP 最小化):\n TN={tn_n} | FP={fp_n}\n FN={fn_n} | TP={tp_n}")

# 5. 繪製新 ROC 曲線
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot(fpr, tpr, color='green', lw=2, label=f'Test ROC (AUC = {auc_new:.2f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 找出 ROC 曲線上對應新閾值的點
idx_new = np.argmin(np.abs(thresholds - NEW_THRESHOLD))
ax3.scatter(fpr[idx_new], tpr[idx_new], s=100, c='red', label=f'Threshold = {NEW_THRESHOLD}')
ax3.scatter(fpr[idx_05], tpr[idx_05], s=50, c='black', alpha=0.5, label='Threshold = 0.5 (Ref)')

ax3.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
        title=f'ROC Curve (Test Set - Threshold {NEW_THRESHOLD})',
        xlabel='False Positive Rate', ylabel='True Positive Rate')
ax3.legend(loc="lower right")
plt.show()

print("=" * 60)