import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ===== 1. 讀取資料 =====
# 請確認你的檔案路徑是否正確，若有報錯請改回絕對路徑
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
target = "Outcome"

# ===== 2. 分離特徵與目標變數 =====
X = df.drop(columns=[target])
y = df[target]

# ===== 3. 切分訓練集與測試集 =====
# 使用 stratify=y 確保訓練集和測試集的比例跟原始資料一樣
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# ★ [新增] 3.5 統計並印出訓練集的類別分佈
# ==========================================
train_counts = y_train.value_counts().sort_index() # 確保 0 在前 1 在後
train_total = len(y_train)

print("\n[統計] 訓練集 (Training Set) 類別分佈狀況:")
print("=" * 50)
print(f"訓練集總筆數: {train_total}")

# 取得 0 和 1 的數量 (使用 .get 以防萬一某類別完全沒出現)
count_0 = train_counts.get(0, 0)
count_1 = train_counts.get(1, 0)

# 計算比例
ratio_0 = count_0 / train_total
ratio_1 = count_1 / train_total

print(f"類別 0 (沒病): {count_0:<5} 筆 | 占比: {ratio_0:.2%}")
print(f"類別 1 (有病): {count_1:<5} 筆 | 占比: {ratio_1:.2%}")
print("-" * 50)

# 簡單判斷不平衡程度
imbalance_ratio = count_0 / count_1 if count_1 > 0 else 0
print(f" 資料不平衡比例 (0 vs 1) 約為: {imbalance_ratio:.1f} : 1")
if imbalance_ratio > 3:
    print("   (警告: 資料嚴重不平衡，建議使用 class_weight='balanced' 或 SMOTE)")
else:
    print("   (資料分佈尚可，通常不需要激進的平衡手段)")
print("=" * 50)


# ===== 4. 設定參數網格 =====
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced', None]
}

# 建立基礎模型
rf = RandomForestClassifier(random_state=42)

# ===== 5. 先跑一次 GridSearch 找出最佳參數 =====
print("\n正在尋找最佳參數中...")
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"最佳參數: {grid_search.best_params_}")
print("-" * 60)

# ===== 5.5 計算測試集混淆矩陣 (純數值) =====
print("\n[新增] 測試集 (Hold-out Test Set) 混淆矩陣數據:")

y_pred_test = best_rf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

print(f"{'類別':<15} | {'預測: 0 (沒病)':<15} | {'預測: 1 (有病)':<15}")
print("-" * 50)
print(f"{'實際: 0 (沒病)':<15} | {tn:<15} | {fp:<15} (誤判有病)")
print(f"{'實際: 1 (有病)':<15} | {fn:<15} | {tp:<15} (抓出有病)")
print("-" * 50)
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive (TP): {tp}")
print("-" * 60)

# ===== 6. 核心步驟：執行 5 折交叉驗證 =====
scoring_metrics = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'auc': 'roc_auc'
}

print("\n🚀 開始執行 5 折交叉驗證 (詳細數據分析)...")

cv_results = cross_validate(
    best_rf, 
    X_train, 
    y_train, 
    cv=5, 
    scoring=scoring_metrics,
    return_train_score=True,
    n_jobs=-1
)

# ===== 7. 定義輸出格式函式 =====
def print_custom_format(set_name, acc_list, auc_list, f1_list):
    print(f"\n===== {set_name} Set 5-Fold CV =====")
    
    for i in range(5):
        print(f"Fold {i+1}: Accuracy={acc_list[i]:.4f}, AUC={auc_list[i]:.4f}, F1={f1_list[i]:.4f}")
    
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
    
    print(f"{set_name} Set Average: Accuracy={acc_mean:.4f} ± {acc_std:.4f}, AUC={auc_mean:.4f} ± {auc_std:.4f}, F1={f1_mean:.4f} ± {f1_std:.4f}")

# ===== 8. 輸出結果 =====
print_custom_format("Training", cv_results['train_accuracy'], cv_results['train_auc'], cv_results['train_f1'])
print_custom_format("Test", cv_results['test_accuracy'], cv_results['test_auc'], cv_results['test_f1'])

print("\n" + "="*50)