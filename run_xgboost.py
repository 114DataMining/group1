import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, confusion_matrix, 
                             roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

# 設定不彈出視窗
import matplotlib
matplotlib.use('Agg')

# ===============================
# 1. 讀取資料
# ===============================
try:
    df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
    print(f"✅ 資料讀取成功！樣本總數: {len(df)}")
except FileNotFoundError:
    print("❌ 錯誤：找不到檔案。")

target = "Outcome"
X = df.drop(columns=[target])
y = df[target]

# ===============================
# 2. 切分訓練集與測試集
# ===============================
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ===============================
# 3. Grid Search
# ===============================
param_grid = {
    'max_depth': [2, 3],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 150],
    'subsample': [0.7, 0.8],
    'gamma': [1, 2]
}

xgb_base = xgb.XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=-1, eval_metric='logloss')
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n===== 正在進行 Grid Search 優化參數... =====")
grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, scoring='roc_auc', cv=cv_strategy, n_jobs=-1)
grid_search.fit(X_train_all, y_train_all)
best_params = grid_search.best_params_
print(f"最佳參數組合: {best_params}")

# ===============================
# 4. 執行 5-Fold CV (加入 SMOTE 並修正表格對齊)
# ===============================
plt.figure(figsize=(10, 8))
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

smote = SMOTE(random_state=42)
tprs = []
mean_fpr = np.linspace(0, 1, 100)

# 表格標題：統一寬度並對齊
print("\n" + "="*115)
header = f"{'折數':<4} | {'Train(0/1)':<13} | {'Val(0/1)':<11} | {'Train-AUC':<7} | {'Train-Acc':<7} | {'Train-F1_score':<7} | {'Val-AUC':<7} | {'Val-Acc':<7} | {'Val-F1_score':<7}"
print(header)
print("-" * 115)

for i, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train_all, y_train_all)):
    X_tr_fold, X_va_fold = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
    y_tr_fold, y_va_fold = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]
    
    # --- SMOTE 處理 ---
    X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr_fold, y_tr_fold)
    
    model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    model.fit(X_tr_resampled, y_tr_resampled)
    
    # 預測機率與標籤
    y_prob_tr = model.predict_proba(X_tr_resampled)[:, 1]
    y_prob_va = model.predict_proba(X_va_fold)[:, 1]
    y_pred_tr = (y_prob_tr > 0.5).astype(int)
    y_pred_va = (y_prob_va > 0.5).astype(int)
    
    # 計算指標
    auc_tr = roc_auc_score(y_tr_resampled, y_prob_tr)
    acc_tr = accuracy_score(y_tr_resampled, y_pred_tr)
    f1_tr = f1_score(y_tr_resampled, y_pred_tr)
    
    auc_va = roc_auc_score(y_va_fold, y_prob_va)
    acc_va = accuracy_score(y_va_fold, y_pred_va)
    f1_va = f1_score(y_va_fold, y_pred_va)
    
    # 統計樣本數
    tr_counts = f"{sum(y_tr_resampled==0)}/{sum(y_tr_resampled==1)}"
    va_counts = f"{sum(y_va_fold==0)}/{sum(y_va_fold==1)}"
    
    # 格式化輸出資料行 (對應標題順序)
    print(f"Fold {i+1:<1} | {tr_counts:<10} (S) | {va_counts:<11} | {auc_tr:.3f} | {acc_tr:.3f} | {f1_tr:.3f} | {auc_va:.3f} | {acc_va:.3f} | {f1_va:.3f}")
    
    fpr, tpr, _ = roc_curve(y_va_fold, y_prob_va)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} Val AUC: {auc_va:.3f}')

print("="*115)

# 繪製平均 ROC
mean_tpr = np.mean(tprs, axis=0)
plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean CV ROC (AUC = {auc(mean_fpr, mean_tpr):.3f})', lw=2)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title('XGBoost 5-Fold ROC (SMOTE後)')
plt.legend()
plt.savefig("roc_smote_analysis.png", dpi=300)
plt.close()

# ===============================
# 5. 繪製 Precision-Recall vs Threshold 曲線 (修正交叉圖)
# ===============================
# 重新用 SMOTE 全訓練集訓練最終模型
X_resampled_all, y_resampled_all = smote.fit_resample(X_train_all, y_train_all)
xgb_final = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
xgb_final.fit(X_resampled_all, y_resampled_all)

y_test_prob = xgb_final.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_prob)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision (精準率)", lw=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall (召回率)", lw=2)
plt.xlabel("Threshold (判別閾值)")
plt.ylabel("Score")
plt.title("XGBoost Precision 與 Recall 隨閾值變動圖 (Test Set)")
plt.axvline(x=0.5, color='red', linestyle=':', label='預設閾值 0.5')
plt.legend()
plt.grid(True)
plt.savefig("precision_recall_threshold.png", dpi=300)
plt.close()

# ===============================
# 6. 最終效能報表
# ===============================
y_test_pred = (y_test_prob > 0.5).astype(int)
print("\n" + "="*40)
print(f" 最終測試集效能報告 (SMOTE 後)")
print(f"測試集 AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"測試集 Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"測試集 F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print("="*40)
print("\n✅ ROC 圖已存至: roc_smote_analysis.png")
print("✅ 閾值交叉圖已存至: precision_recall_threshold.png")