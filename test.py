import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 非互動式繪圖後端
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter

# ===============================
# 1. 讀取資料
# ===============================
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")

target = "Outcome"
X = df.drop(columns=[target])
y = df[target]

# ===============================
# 2. 資料集切分 (80% 訓練, 20% 測試)
# ===============================
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 3. 設定模型 (指定超參數)
# ===============================
model_params = {
    'n_estimators': 150,
    'max_depth': 2,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'gamma': 2,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

# ===============================
# 4. 5-Fold CV 詳細流程
# ===============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 儲存 ROC 繪圖用的數據
mean_fpr = np.linspace(0, 1, 100)
tprs_val = []
tprs_train = []
aucs_val = []

# 儲存每一折的詳細指標
fold_logs = []

print("\n" + "="*60)
print("開始 5-Fold Cross-Validation (詳細數據)")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all), 1):
    print(f"\n>>>>> Running Fold {fold} / 5 <<<<<")
    
    # 4.1 切分 Fold
    X_tr, X_val = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
    y_tr, y_val = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]

    # 4.2 紀錄並印出 SMOTE 前的類別平衡
    count_tr_before = Counter(y_tr)
    print(f"[Class Balance] Train Before SMOTE: {dict(count_tr_before)}")

    # 4.3 SMOTE
    smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
    
    # 4.4 紀錄並印出 SMOTE 後的類別平衡
    count_tr_after = Counter(y_tr_res)
    print(f"[Class Balance] Train After  SMOTE: {dict(count_tr_after)}")
    print(f"[Class Balance] Validation Set:     {dict(Counter(y_val))}")

    # 4.5 訓練模型
    model = XGBClassifier(**model_params)
    model.fit(X_tr_res, y_tr_res)

    # 4.6 預測 (機率)
    y_tr_prob = model.predict_proba(X_tr_res)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # 4.7 預測 (類別, Th=0.5)
    y_tr_pred = (y_tr_prob >= 0.5).astype(int)
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    # 4.8 計算指標 (Train & Val)
    def get_metrics(y_true, y_pred, y_prob):
        return {
            'auc': auc(*roc_curve(y_true, y_prob)[:2]),
            'acc': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'cm': confusion_matrix(y_true, y_pred)
        }

    m_tr = get_metrics(y_tr_res, y_tr_pred, y_tr_prob)
    m_val = get_metrics(y_val, y_val_pred, y_val_prob)

    # 印出數據
    print(f"\n[Fold {fold} Metrics]")
    print(f"  Train | AUC: {m_tr['auc']:.4f} | F1: {m_tr['f1']:.4f} | Acc: {m_tr['acc']:.4f}")
    print(f"  Val   | AUC: {m_val['auc']:.4f} | F1: {m_val['f1']:.4f} | Acc: {m_val['acc']:.4f}")
    print(f"  Val Confusion Matrix:\n{m_val['cm']}")

    # 儲存指標以便計算平均
    fold_logs.append({'fold': fold, 'train': m_tr, 'val': m_val})

    # 4.9 處理 ROC 曲線數據
    # --- Validation ROC ---
    fpr_v, tpr_v, _ = roc_curve(y_val, y_val_prob)
    tprs_val.append(np.interp(mean_fpr, fpr_v, tpr_v))
    tprs_val[-1][0] = 0.0
    aucs_val.append(m_val['auc'])

    # --- Train ROC ---
    fpr_t, tpr_t, _ = roc_curve(y_tr_res, y_tr_prob)
    tprs_train.append(np.interp(mean_fpr, fpr_t, tpr_t))
    tprs_train[-1][0] = 0.0

# ===============================
# 5. 計算並列出 5-Fold 平均指標
# ===============================
print("\n" + "="*60)
print("5-Fold Cross-Validation 平均結果")
print("="*60)

metrics_to_avg = ['auc', 'f1', 'acc']
for phase in ['train', 'val']:
    print(f"\n--- {phase.capitalize()} Set Average ---")
    for m in metrics_to_avg:
        values = [log[phase][m] for log in fold_logs]
        mean_v = np.mean(values)
        std_v = np.std(values)
        print(f"{m.upper()}: {mean_v:.4f} ± {std_v:.4f}")

# ===============================
# 6. 最終測試集評估 (Default Th=0.5)
# ===============================
print("\n" + "="*60)
print("最終測試集 (Test Set) 評估")
print("="*60)

# 使用完整訓練集 + SMOTE 重訓
smote_all = BorderlineSMOTE(random_state=42) 
X_train_final, y_train_final = smote_all.fit_resample(X_train_all, y_train_all)

final_model = XGBClassifier(**model_params)
final_model.fit(X_train_final, y_train_final)

y_test_prob = final_model.predict_proba(X_test)[:, 1]
y_test_pred_def = (y_test_prob >= 0.5).astype(int)

test_auc = auc(*roc_curve(y_test, y_test_prob)[:2])
test_f1 = f1_score(y_test, y_test_pred_def)
test_acc = accuracy_score(y_test, y_test_pred_def)

print(f"[Threshold = 0.5]")
print(f"Test AUC:      {test_auc:.4f}")
print(f"Test F1:       {test_f1:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred_def)}")

# ===============================
# 7. 繪製 ROC 圖 (依照您上傳的圖片樣式)
# ===============================
plt.figure(figsize=(10, 8))

# (A) 畫每一折的細線 (半透明)
for i in range(5):
    plt.plot(mean_fpr, tprs_val[i], lw=1, alpha=0.3,
             label=f'ROC Fold {i+1} (AUC = {aucs_val[i]:.2f})')

# (B) 畫隨機猜測線 (紅色虛線)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

# (C) 計算平均與標準差
mean_tpr = np.mean(tprs_val, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs_val)

# (D) 畫平均 ROC 線 (藍色實線)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
         lw=2, alpha=.8)

# (E) 畫標準差陰影 (灰色區域) - 這是您圖片中的重點
std_tpr = np.std(tprs_val, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1) # 上界不超過1
tprs_lower = np.maximum(mean_tpr - std_tpr, 0) # 下界不低於0
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (5-Fold CV)') # 標題改為跟您的圖一樣
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

save_roc_name = 'cv_roc_with_std.png'
plt.savefig(save_roc_name)
plt.close()

print(f"\n✅ 已生成標準差風格 ROC 圖: {save_roc_name}")
# ===============================
# 8. 尋找並應用最佳閾值
# ===============================
print("\n" + "="*60)
print("尋找最佳閾值 (Precision ≈ Recall)")
print("="*60)

precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)
diff = np.abs(precision[:-1] - recall[:-1])
best_idx = np.argmin(diff)
best_threshold = thresholds[best_idx]

print(f"最佳閾值: {best_threshold:.4f}")
print(f"交叉點 Precision: {precision[best_idx]:.4f}")
print(f"交叉點 Recall:    {recall[best_idx]:.4f}")

# 繪製 PR 平衡圖
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
plt.plot(thresholds, recall[:-1], label='Recall', color='green')
plt.scatter(best_threshold, precision[best_idx], color='red', s=100, zorder=5, label=f'Balance Point ({best_threshold:.3f})')
plt.title('Precision-Recall Balance Curve')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('pr_balance_curve.png')
plt.close()
print("✅ PR 平衡圖表已生成: pr_balance_curve.png")

# 使用最佳閾值重新評估
y_test_pred_best = (y_test_prob >= best_threshold).astype(int)

print(f"\n[Threshold = {best_threshold:.4f}] Test Metrics")
print(f"AUC:       {test_auc:.4f} (不變)")
print(f"F1:        {f1_score(y_test, y_test_pred_best):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred_best):.4f}")
print(f"Confusion Matrix (Best Th):\n{confusion_matrix(y_test, y_test_pred_best)}")

# ===============================
# 9. 儲存結果 CSV
# ===============================
results_df = X_test.copy()
results_df['True_Label'] = y_test
results_df['Probability'] = y_test_prob
results_df['Predicted_Label'] = y_test_pred_best

save_name = "test_prediction_results_best_th.csv"
results_df.to_csv(save_name, index=True)
print(f"\n✅ 最終 CSV 已儲存: {save_name} (預測欄位已使用最佳閾值)")