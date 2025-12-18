import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_curve
)

# ===== 1. 讀取資料 =====
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
target = "Outcome"  # 目標變數：0=沒有糖尿病, 1=有糖尿病

# ===== 2. 分離特徵與目標變數 =====
X = df.drop(columns=[target])  # 特徵（所有輸入變數）
y = df[target]                  # 目標（要預測的結果）

# ===== 3. 切分訓練集與測試集 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80%訓練，20%測試
    random_state=42,    # 固定亂數種子，確保結果可重現
    stratify=y          # 保持訓練集與測試集中正負樣本的比例一致
)

print(f"訓練集大小: {len(X_train)}, 測試集大小: {len(X_test)}")
print(f"訓練集中 Outcome=1 的比例: {y_train.sum()/len(y_train):.2%}")

# ===== 4. 設定參數網格（用於尋找最佳參數組合）=====
param_grid = {
    'n_estimators':[50,80,100,150],      # 增加樹的數量
    'max_depth': [4, 5, 6, 7, 8],            # 改為保守範圍
    'min_samples_leaf': [2, 4, 8],        # 提高最小值
    'min_samples_split': [10, 20],        # 新增：控制分裂條件
    'class_weight': ['balanced', {0:1, 1:2}]
}

# ===== 5. 建立基礎模型 =====
rf = RandomForestClassifier(random_state=42)

# ===== 6. 使用 GridSearchCV 尋找最佳參數 =====
print("\n開始尋找最佳參數組合...")
grid_search = GridSearchCV(
    estimator=rf,           # 要優化的模型
    param_grid=param_grid,  # 參數搜尋範圍
    cv=5,                   # 5折交叉驗證（將訓練集分5份，輪流當驗證集）
    scoring='f1',           # 優化目標：F1-Score（Precision與Recall的調和平均）
    n_jobs=-1,              # 使用所有CPU核心加速
    verbose=1               # 顯示進度
)

grid_search.fit(X_train, y_train)

# ===== 7. 取得最佳模型 =====
best_rf = grid_search.best_estimator_
print(f"\n✅ 最佳參數組合: {grid_search.best_params_}")
print(f"✅ 訓練時最佳 F1-Score: {grid_search.best_score_:.4f}")

# ===== 8. 使用最佳模型進行預測 =====
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]  # 預測為1的機率

# 預設閾值 0.5 的預測結果
y_pred_default = best_rf.predict(X_test)

# 自訂閾值 0.6 的預測結果
custom_threshold = 0.6
y_pred_custom = (y_pred_prob >= custom_threshold).astype(int)

# ===== 9. 計算兩種閾值的評估指標 =====
print("\n" + "="*60)
print("📊 Random Forest 模型評估結果比較")
print("="*60)

# --- 預設閾值 0.5 ---
print("\n【預設閾值 0.5】")
accuracy_default = accuracy_score(y_test, y_pred_default)
precision_default = precision_score(y_test, y_pred_default, pos_label=1)
recall_default = recall_score(y_test, y_pred_default, pos_label=1)
f1_default = f1_score(y_test, y_pred_default, pos_label=1)

print(f"Accuracy  (準確率):   {accuracy_default:.4f}")
print(f"Precision (精確率):   {precision_default:.4f}  ← 預測為「有病」中真的有病的比例")
print(f"Recall    (召回率):   {recall_default:.4f}  ← 實際有病的人中被找出來的比例")
print(f"F1-Score  (F1分數):   {f1_default:.4f}")

cm_default = confusion_matrix(y_test, y_pred_default)
print("\n混淆矩陣:")
print("           預測: 0    預測: 1")
print(f"實際: 0   {cm_default[0,0]:4d}      {cm_default[0,1]:4d}   (TN / FP)")
print(f"實際: 1   {cm_default[1,0]:4d}      {cm_default[1,1]:4d}   (FN / TP)")

# --- 自訂閾值 0.6 ---
print("\n" + "="*60)
print("【自訂閾值 0.6】（提高精確率，降低誤報）")
accuracy_custom = accuracy_score(y_test, y_pred_custom)
precision_custom = precision_score(y_test, y_pred_custom, pos_label=1)
recall_custom = recall_score(y_test, y_pred_custom, pos_label=1)
f1_custom = f1_score(y_test, y_pred_custom, pos_label=1)

print(f"Accuracy  (準確率):   {accuracy_custom:.4f}")
print(f"Precision (精確率):   {precision_custom:.4f}  ← 預測為「有病」中真的有病的比例 ⬆️")
print(f"Recall    (召回率):   {recall_custom:.4f}  ← 實際有病的人中被找出來的比例 ⬇️")
print(f"F1-Score  (F1分數):   {f1_custom:.4f}")

cm_custom = confusion_matrix(y_test, y_pred_custom)
print("\n混淆矩陣:")
print("           預測: 0    預測: 1")
print(f"實際: 0   {cm_custom[0,0]:4d}      {cm_custom[0,1]:4d}   (TN / FP)")
print(f"實際: 1   {cm_custom[1,0]:4d}      {cm_custom[1,1]:4d}   (FN / TP)")

# AUC 不受閾值影響
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC (曲線下面積): {auc:.4f}  ← 不受閾值影響")

# ===== 10. 完整分類報告（使用閾值 0.6）=====
print("\n" + "="*60)
print("📄 完整分類報告（閾值 0.6）:")
print("="*60)
print(classification_report(y_test, y_pred_custom, target_names=['無糖尿病(0)', '有糖尿病(1)']))

# ===== 11. 繪製 ROC 曲線 =====
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# 標記閾值 0.5 和 0.6 的位置
idx_05 = np.argmin(np.abs(thresholds - 0.5))
idx_07 = np.argmin(np.abs(thresholds - 0.6))
plt.scatter(fpr[idx_05], tpr[idx_05], color='blue', s=100, zorder=5, label='Threshold = 0.5')
plt.scatter(fpr[idx_07], tpr[idx_07], color='red', s=100, zorder=5, label='Threshold = 0.6')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (假陽性率)', fontsize=12)
plt.ylabel('True Positive Rate (真陽性率/Recall)', fontsize=12)
plt.title('ROC Curve - Random Forest 糖尿病預測模型', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✨ 分析完成！")
print("\n💡 解讀:")
print("   • 閾值從 0.5 提高到 0.6 後：")
print("   • Precision ⬆️ (精確率提高 - 減少誤報，預測有病時更可靠)")
print("   • Recall ⬇️ (召回率降低 - 漏掉一些真正有病的患者)")
print("   • 適用場景：希望減少「誤診為有病」的情況時使用")