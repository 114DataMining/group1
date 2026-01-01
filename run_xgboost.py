import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 設定不彈出視窗，直接繪圖至後台
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
# 2. 切分訓練集 (80%, 約592筆) 與測試集 (20%, 約148筆)
# ===============================
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y,
    test_size=0.20, 
    stratify=y,
    random_state=42
)

# ===============================
# 3. 執行 Grid Search 優化
# ===============================
param_grid = {
    'max_depth': [2, 3],
    'min_child_weight': [10, 11, 12],
    'learning_rate': [0.01, 0.05, 0.08],
    'subsample': [0.6, 0.7, 0.8],
    'gamma': [1, 2],
    'reg_lambda': [5, 10],
    'n_estimators': [100, 150, 200]
}

scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'F1': 'f1'}
ratio = sum(y_train_all == 0) / sum(y_train_all == 1)

xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n===== 正在進行 Grid Search 優化參數... =====")
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring=scoring,
    refit='AUC',
    cv=cv_strategy,
    n_jobs=-1,
    return_train_score=True
)
grid_search.fit(X_train_all, y_train_all)

best_params = grid_search.best_params_
print(f"\n🏆 最佳參數組合: {best_params}")

# ===============================
# 4. 繪製 5-Fold ROC 並統計類別數量
# ===============================
plt.figure(figsize=(10, 8))
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

tprs = []
mean_fpr = np.linspace(0, 1, 100)

print("\n" + "="*60)
print(f"{'折數':<4} | {'訓練集 (0/1)':<15} | {'驗證集 (0/1)':<15} | {'Train AUC':<10} | {'Val AUC':<10}")
print("-" * 60)

for i, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train_all, y_train_all)):
    X_tr, X_va = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
    y_tr, y_va = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]
    
    # 統計類別個數
    tr_0, tr_1 = (y_tr == 0).sum(), (y_tr == 1).sum()
    va_0, va_1 = (y_va == 0).sum(), (y_va == 1).sum()
    
    model = xgb.XGBClassifier(**best_params, scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
    model.fit(X_tr, y_tr)
    
    # 指標計算
    y_prob_va = model.predict_proba(X_va)[:, 1]
    fpr, tpr, _ = roc_curve(y_va, y_prob_va)
    roc_auc_va = auc(fpr, tpr)
    
    y_prob_tr = model.predict_proba(X_tr)[:, 1]
    roc_auc_tr = roc_auc_score(y_tr, y_prob_tr)
    
    # 輸出格式化結果
    print(f"Fold {i+1:<1} | {tr_0:>3}/{tr_1:<3}        | {va_0:>3}/{va_1:<3}        | {roc_auc_tr:.4f}    | {roc_auc_va:.4f}")
    
    plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f'Fold {i+1} Val (AUC = {roc_auc_va:.3f})')
    
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

print("="*60)

# 繪製平均線與測試集
mean_tpr = np.mean(tprs, axis=0)
plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean CV Val ROC (AUC = {auc(mean_fpr, mean_tpr):.3f})', lw=2)

xgb_final = grid_search.best_estimator_
y_test_prob = xgb_final.predict_proba(X_test)[:, 1]
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
plt.plot(fpr_test, tpr_test, color='red', label=f'Final Test ROC (AUC = {auc(fpr_test, tpr_test):.3f})', lw=3, linestyle='--')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('XGBoost 訓練進度與樣本分佈分析')
plt.legend(loc="lower right")
plt.savefig("roc_with_class_counts.png", dpi=300)
plt.close()

# ===============================
# 5. 輸出最終效能報表
# ===============================
best_idx = grid_search.best_index_
res = grid_search.cv_results_

# 最終測試集統計
te_0, te_1 = (y_test == 0).sum(), (y_test == 1).sum()
y_test_pred = (y_test_prob > 0.5).astype(int)

print("\n" + "="*40)
print(f"📊 最終測試集樣本分佈 (n={len(y_test)})")
print(f"0 (健康): {te_0} 筆 | 1 (患病): {te_1} 筆")
print("-" * 40)
print(f"測試集 AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"測試集 Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"測試集 F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print("="*40)

print("\n統計圖表與數據已存至: roc_with_class_counts.png")
# ===============================
# 6. 特徵重要性分析 (Feature Importance)
# ===============================
# 取得特徵名稱與分數
importances = xgb_final.feature_importances_
feature_names = X.columns

# 建立 DataFrame 並排序
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True) # 由小到大排，方便橫向圖由上到下顯示

# 繪圖
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('XGBoost 特徵重要性分析 (Feature Importance)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 在長條圖上標註數值
for index, value in enumerate(feature_importance_df['Importance']):
    plt.text(value, index, f'{value:.4f}')

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("\n特徵重要性圖表已存至: feature_importance.png")

# 額外輸出文字版清單（由高到低）
print("\n特徵重要性排名:")
print("-" * 30)
print(feature_importance_df.sort_values(by='Importance', ascending=False).to_string(index=False))
print("-" * 30)