import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # [新增] 用來顯示檔案路徑
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
    'rf__n_estimators': [100, 200, 300],        # 樹的數量
    'rf__max_depth': [5, 10, None],             # 樹的深度
    'rf__min_samples_split': [2, 5],            # 節點分裂所需最小樣本數
    'rf__min_samples_leaf': [1, 2, 4],          # 葉子節點最少樣本數
    'rf__max_features': ['sqrt', 'log2']        # 最大特徵數選擇
}

# 執行搜尋 (修正: 改回 f1 以針對不平衡資料)
grid_search = GridSearchCV(
    estimator=pipeline_grid,
    param_grid=param_grid,
    cv=5, 
    scoring='roc_auc',  # ★ 這裡幫你改回來了，原本 'None' 會報錯
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 列出所有超參數配對結果
print("\n[詳細報告] 每個超參數組合的測試結果 (按 F1 分數排序):")
results_df = pd.DataFrame(grid_search.cv_results_)
cols_to_show = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
print(results_df[cols_to_show].sort_values(by='rank_test_score').to_string(index=False))

# 取得最佳參數與模型
best_params = grid_search.best_params_
best_model_pipeline = grid_search.best_estimator_

print(f"\n最終選定最佳參數: {best_params}")
print(f"最佳 CV F1 Score: {grid_search.best_score_:.4f}")

# ==========================================
# 4. 使用「最佳參數」執行詳細的 5-Fold CV (含訓練集與驗證集比較)
# ==========================================
print("\n" + "=" * 80)
print("[3] 針對「最佳參數」執行 5-Fold CV 詳細分析 (Train vs Validation)")
print("=" * 80)

# 初始化儲存指標的 List (訓練集用)
tr_accs, tr_f1s, tr_aucs = [], [], []
# 初始化儲存指標的 List (驗證集用)
val_accs, val_f1s, val_aucs = [], [], []

# 初始化畫圖用的變數
tprs = []
mean_fpr = np.linspace(0, 1, 100)
fig1, ax1 = plt.subplots(figsize=(10, 8))

# 設定 K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    # 1. 切分資料
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # 2. 顯示 SMOTE 平衡前後的 0/1 數量 (僅供顯示用，實際平衡由 Pipeline 處理)
    # 為了顯示數量，我們在這裡手動做一次 SMOTE check
    smote_check = BorderlineSMOTE(random_state=42, kind='borderline-1')
    X_tr_res_check, y_tr_res_check = smote_check.fit_resample(X_tr, y_tr)
    
    print(f"\n🔹 [Fold {fold}] 數據分佈詳情:")
    print(f"   【平衡前】 0 (沒病): {sum(y_tr==0)} | 1 (有病): {sum(y_tr==1)}")
    print(f"   【平衡後】 0 (沒病): {sum(y_tr_res_check==0)} | 1 (有病): {sum(y_tr_res_check==1)}")

    # 3. 訓練模型 (Pipeline 會自動處理 Scaling -> SMOTE -> RF)
    best_model_pipeline.fit(X_tr, y_tr)

    # 4. === 計算 [訓練集 Training Set] 指標 (自己考自己) ===
    y_tr_pred = best_model_pipeline.predict(X_tr)
    y_tr_prob = best_model_pipeline.predict_proba(X_tr)[:, 1]
    
    tr_acc = accuracy_score(y_tr, y_tr_pred)
    tr_f1 = f1_score(y_tr, y_tr_pred)
    tr_auc = roc_auc_score(y_tr, y_tr_prob)
    
    tr_accs.append(tr_acc)
    tr_f1s.append(tr_f1)
    tr_aucs.append(tr_auc)

    # 5. === 計算 [驗證集 Validation Set] 指標 (模擬考) ===
    y_val_pred = best_model_pipeline.predict(X_val)
    y_val_prob = best_model_pipeline.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    val_aucs.append(val_auc)

    # 6. 印出該折結果比較
    print(f"   📊 指標比較:")
    print(f"      Train (訓練): Acc={tr_acc:.4f} | F1={tr_f1:.4f} | AUC={tr_auc:.4f}")
    print(f"      Valid (驗證): Acc={val_acc:.4f} | F1={val_f1:.4f} | AUC={val_auc:.4f}")

    # 7. 畫 ROC 線 (只畫驗證集的)
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    ax1.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold} (AUC = {val_auc:.2f})')
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

# --- 繪圖收尾 ---
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(val_aucs)
ax1.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} +/- {std_auc:.2f})', lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='+/- 1 std. dev.')

ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve (5-Fold CV - Validation)", xlabel='False Positive Rate', ylabel='True Positive Rate')
ax1.legend(loc="lower right")
plt.show()

# --- 最終統計表格 ---
print("\n" + "="*60)
print("             【5-Fold CV 最終平均結果統計】")
print("="*60)
print(f"{'Metric':<10} | {'Training Set (Mean ± Std)':<25} | {'Validation Set (Mean ± Std)':<25}")
print("-" * 65)
print(f"{'Accuracy':<10} | {np.mean(tr_accs):.4f} ± {np.std(tr_accs):.4f}      | {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
print(f"{'F1 Score':<10} | {np.mean(tr_f1s):.4f} ± {np.std(tr_f1s):.4f}      | {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
print(f"{'AUC':<10}      | {np.mean(tr_aucs):.4f} ± {np.std(tr_aucs):.4f}      | {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
print("="*60)
print("💡 觀察重點：如果 Training 分數遠高於 Validation 分數 (例如差 0.1 以上)，")
print("             則模型可能存在過擬合 (Overfitting) 現象。")

# ==========================================
# 5. 最終模型訓練與標準測試 (Threshold = 0.5)
# ==========================================
print("\n" + "-" * 60)
print("[4] 最終模型訓練與標準測試 (Threshold = 0.5)")
print("-" * 60)

# 1. 檢查最終訓練集 SMOTE 後的數量
print("正在對完整 80% 訓練集進行 SMOTE 數量檢查...")
smote_final_check = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_final_res, y_final_res = smote_final_check.fit_resample(X_train, y_train)
print(f"   原始訓練集分佈: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
print(f"   平衡後訓練集分佈: 0={sum(y_final_res==0)}, 1={sum(y_final_res==1)}")

# 2. 訓練最終模型 (使用最佳參數)
best_model_pipeline.fit(X_train, y_train)

# 3. 預測測試集 (產生機率值)
y_prob_test = best_model_pipeline.predict_proba(X_test)[:, 1]

# 4. 產生標準預測結果 (閾值 0.5)
y_pred_std = best_model_pipeline.predict(X_test)

# 5. 計算標準指標
acc_std = accuracy_score(y_test, y_pred_std)
f1_std = f1_score(y_test, y_pred_std)
auc_std = roc_auc_score(y_test, y_prob_test)

print(f"\n[標準測試集結果 (Threshold = 0.5)]")
print(f"Accuracy: {acc_std:.4f}")
print(f"AUC:      {auc_std:.4f}")
print(f"F1 Score: {f1_std:.4f}")

# 6. 標準混淆矩陣
cm_std = confusion_matrix(y_test, y_pred_std)
tn, fp, fn, tp = cm_std.ravel()
print(f"混淆矩陣:\n TN={tn} | FP={fp}\n FN={fn} | TP={tp}")

# 7. 繪製標準 ROC 曲線
fig2, ax2 = plt.subplots(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'Test ROC (AUC = {auc_std:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# 標記 0.5 的位置
idx_05 = np.argmin(np.abs(thresholds - 0.5))
ax2.scatter(fpr[idx_05], tpr[idx_05], s=100, c='black', label='Threshold = 0.5')

ax2.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
        title='ROC Curve (Test Set - Standard)',
        xlabel='False Positive Rate', ylabel='True Positive Rate')
ax2.legend(loc="lower right")
plt.show()

# ==========================================
# 6. 匯出機率 CSV 與 閾值調整測試 (Threshold = 0.65)
# ==========================================
print("\n" + "-" * 60)
print("[5] 匯出 CSV 與 閾值調整分析 (Threshold = 0.65)")
print("-" * 60)

# 1. 製作 DataFrame 並存檔
df_result = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted_Probability': y_prob_test,
    'Pred_0.5': y_pred_std
})
csv_name = "diabetes_test_probabilities.csv"
df_result.to_csv(csv_name, index=False)

# ★ 這裡會告訴你檔案在哪裡
print(f"✅ CSV 檔案已成功建立！")
print(f"📂 檔案名稱: {csv_name}")
print(f"📍 完整路徑: {os.path.abspath(csv_name)}")

# 2. 應用新的閾值 (0.65)
NEW_THRESHOLD = 0.65
# 邏輯：只有機率 >= 0.65 才是 1，否則為 0
y_pred_new = (y_prob_test >= NEW_THRESHOLD).astype(int)

# 3. 計算新指標
acc_new = accuracy_score(y_test, y_pred_new)
f1_new = f1_score(y_test, y_pred_new)
# 注意：AUC 不會因為閾值改變而改變，因為 AUC 是看整體排序
# 但為了完整性我們還是印出來 (數值會跟上面一樣)
auc_new = roc_auc_score(y_test, y_prob_test) 

print(f"\n[調整後測試集結果 (Threshold = {NEW_THRESHOLD})]")
print(f"Accuracy: {acc_new:.4f}")
print(f"AUC:      {auc_new:.4f} (AUC與閾值無關，數值不變)")
print(f"F1 Score: {f1_new:.4f}")

# 4. 新混淆矩陣
cm_new = confusion_matrix(y_test, y_pred_new)
tn_n, fp_n, fn_n, tp_n = cm_new.ravel()
print(f"新混淆矩陣:\n TN={tn_n} | FP={fp_n} (預期變少)\n FN={fn_n} (預期變多)| TP={tp_n}")

# 5. 繪製新 ROC 曲線 (標示出 0.65 的點)
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot(fpr, tpr, color='green', lw=2, label=f'Test ROC (AUC = {auc_new:.2f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 找出 ROC 曲線上對應 0.65 的點
idx_new = np.argmin(np.abs(thresholds - NEW_THRESHOLD))
ax3.scatter(fpr[idx_new], tpr[idx_new], s=100, c='red', label=f'Threshold = {NEW_THRESHOLD}')

# 把原本 0.5 的點也畫上去做比較
ax3.scatter(fpr[idx_05], tpr[idx_05], s=50, c='black', alpha=0.5, label='Threshold = 0.5 (Ref)')

ax3.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
        title=f'ROC Curve (Test Set - Threshold {NEW_THRESHOLD})',
        xlabel='False Positive Rate', ylabel='True Positive Rate')
ax3.legend(loc="lower right")
plt.show()

print("=" * 60)
