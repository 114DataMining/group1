import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE

# 1. 讀取資料與切分 (80% 訓練, 20% 測試)
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
X, y = df.drop(columns=["Outcome"]), df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("-" * 30)
print(f"原始訓練集分佈: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
print(f"原始測試集分佈: 0={sum(y_test==0)}, 1={sum(y_test==1)}")
print("-" * 30)

# 2. GridSearchCV 尋找最佳超參數
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', BorderlineSMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')) # 加上 class_weight 有助於抑制 FP
])

param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2']
}

print("\n[GridSearch] 執行中...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"最佳參數: {grid_search.best_params_}")

# 3. 使用最佳參數執行詳細 5-Fold 交叉驗證
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {'acc': [], 'f1': [], 'auc': []}
tprs, mean_fpr = [], np.linspace(0, 1, 100)

print("\n" + "="*50 + "\n[詳細 5-Fold CV 報告]")
plt.figure(figsize=(8, 6))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    xt, xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
    yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 這裡手動執行一次 SMOTE 僅為了顯示數量，實際訓練由 pipeline 完成
    sm = BorderlineSMOTE(random_state=42)
    _, yt_res = sm.fit_resample(xt, yt)
    
    # 訓練與預測
    best_model.fit(xt, yt)
    y_pred = best_model.predict(xv)
    y_prob = best_model.predict_proba(xv)[:, 1]
    
    # 指標計算
    m_acc = accuracy_score(yv, y_pred)
    m_f1 = f1_score(yv, y_pred)
    m_auc = roc_auc_score(yv, y_prob)
    for k, v in zip(cv_metrics.keys(), [m_acc, m_f1, m_auc]): cv_metrics[k].append(v)
    
    print(f"Fold {fold}:")
    print(f"  [數量] 前: 0={sum(yt==0)}, 1={sum(yt==1)} | 後: 0={sum(yt_res==0)}, 1={sum(yt_res==1)}")
    print(f"  [表現] Acc={m_acc:.3f}, F1={m_f1:.3f}, AUC={m_auc:.3f}")
    
    fpr, tpr, _ = roc_curve(yv, y_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold} (AUC={m_auc:.2f})')

# 繪製 CV 平均 ROC
mean_tpr = np.mean(tprs, axis=0)
plt.plot(mean_fpr, mean_tpr, 'b', lw=2, label=f'Mean ROC (AUC={np.mean(cv_metrics["auc"]):.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.title("5-Fold Cross Validation ROC")
plt.legend()
plt.show()

# 4. 最終模型訓練與標準測試 (Threshold = 0.5)
print("\n" + "="*50 + "\n[最終測試集結果 - Threshold 0.5]")
# 顯示最終訓練前的 SMOTE 數量
_, y_train_res = sm.fit_resample(X_train, y_train)
print(f"最終訓練 SMOTE 後數量: 0={sum(y_train_res==0)}, 1={sum(y_train_res==1)}")

best_model.fit(X_train, y_train) 
y_final_pred = best_model.predict(X_test)
y_final_prob = best_model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_final_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_final_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_final_prob):.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_final_pred).ravel()
print(f"\n混淆矩陣:\nTN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}")