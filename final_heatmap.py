import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 載入原始數據檔案
file_path = 'diabetes_skinthickness_knn_imputed.csv'

# 確認檔案是否存在
if not os.path.exists(file_path):
    print(f"錯誤：找不到檔案 '{file_path}'。請確認檔案是否在正確的資料夾中。")
else:
    try:
        # 使用 sep=None 和 engine='python' 讓 Pandas 自動偵測是逗號還是 Tab 分隔
        df_original = pd.read_csv(file_path, sep=None, engine='python')
        print("檔案讀取成功！")
        
        # 2. 數據清洗：確保只計算「數值型」欄位
        # 如果欄位包含文字，corr() 可能會報錯或產生非預期結果
        df_numeric = df_original.select_dtypes(include=[np.number])

        # 3. 計算相關係數矩陣
        correlation_matrix = df_numeric.corr(method='pearson')

        # 4. 繪製相關性熱力圖
        plt.figure(figsize=(12, 10)) # 稍微放大尺寸確保文字不重疊
        
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            linewidths=.5, 
            cbar_kws={'label': 'Pearson Correlation Coefficient'},
            square=True # 強制格子成正方形，美觀考量
        )

        plt.title('Correlation Heatmap (Diabetes Dataset)', fontsize=15)
        
        # 解決中文字體或負號顯示問題（若有需要）
        plt.tight_layout() 
        
        # 存檔並顯示
        plt.savefig('final_heatmap.png', dpi=300) # 提高解析度
        plt.show() # 在 Jupyter 或 IDE 中直接彈出圖表
        print("熱力圖已成功生成並儲存為 'final_heatmap.png'")

    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")