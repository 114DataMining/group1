import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 設定字體以支援中文顯示
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 讀取與準備資料
df = pd.read_csv("diabetes_skinthickness_knn_imputed.csv")
target = "Outcome"
X = df.drop(columns=[target])
y = df[target]

# 2. 切分資料並使用 SMOTE 平衡樣本
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. 訓練最終模型
best_params = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 150,
    'subsample': 0.8,
    'gamma': 1
}
model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
model.fit(X_resampled, y_resampled)

# ============================================================
# 4. Tree SHAP 可解釋性分析
# ============================================================
print("\n===== 正在計算 Tree SHAP 解釋值... =====")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- A. 繪製 SHAP Summary Plot (蜂群圖) ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Tree SHAP 特徵影響力分析 (Summary Plot)")
plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=300)
plt.close()

# --- B. 繪製 SHAP Bar Plot (含數值標註) ---
# 計算平均絕對 SHAP 值並排序
if isinstance(shap_values, list): # 針對某些版本的 SHAP 處理多分類輸出格式
    mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

df_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': mean_abs_shap
}).sort_values(by='importance', ascending=True) # 為了橫向圖由大到小顯示，這裡需由小到大排

plt.figure(figsize=(10, 6))
# 繪製長條圖，顏色設定為原本 SHAP 預設的藍色
bars = plt.barh(df_importance['feature'], df_importance['importance'], color='#1E90FF', alpha=0.8)

# 在長條圖末端標註數值
for bar in bars:
    width = bar.get_width()
    plt.text(width + (max(mean_abs_shap)*0.01), # X偏移量
             bar.get_y() + bar.get_height()/2,  # Y位置在長條中間
             f'{width:.4f}',                    # 格式化為小數點四位
             va='center', fontsize=10, fontweight='bold')

plt.title("Tree SHAP 特徵重要性排名 (Bar Plot with Values)")
plt.xlabel("平均絕對 SHAP 值 (mean(|SHAP value|))")
plt.tight_layout()
plt.savefig("shap_bar_plot_with_values.png", bbox_inches='tight', dpi=300)
plt.close()

print("✅ SHAP 分析完成！")
print("✅ 已存檔：shap_summary_plot.png")
print("✅ 已存檔：shap_bar_plot_with_values.png (已標註數值)")

# 額外輸出文字版清單給老師看
print("\n--- 特徵重要性數值清單 ---")
print(df_importance.sort_values(by='importance', ascending=False).to_string(index=False))