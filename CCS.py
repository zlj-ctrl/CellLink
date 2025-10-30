import pandas as pd
import numpy as np
import os
import math


# 定义细胞类型
celltypes =  ["B cells", "Epithelial cells", "Mast cells", 
             "Myeloids", "Stromal cells", "T cells"]


# 创建一个空矩阵来存储结果
result_matrix = pd.DataFrame(np.zeros((len(celltypes), len(celltypes))), index=celltypes, columns=celltypes)

# 文件路径模板
file_template = "D:\\work\\CC_communication\\result\\CRC\\2000\\tumor\\{celltype1}\\{celltype1}_{celltype2}_0.2_3.csv"

# 遍历所有细胞类型组合
for i in range(len(celltypes)):
    for j in range(len(celltypes)):
        celltype1 = celltypes[i]
        celltype2 = celltypes[j]

        # 构建文件路径
        file_path = file_template.format(celltype1=celltype1, celltype2=celltype2)
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            df1 = pd.read_csv(file_path, index_col=0)
            df1_values = df1.values

            # 创建一个与df1相同维度的零矩阵
            df2 = np.zeros(df1_values.shape)

            # 计算二值化后的强度
            positive_count = 0
            intensity_sum = 0
            for m in range(len(df1_values)):
                for n in range(len(df1_values)):
                    if df1_values[m][n] > 0:
                        df2[m][n] = 1
                        positive_count += 1
                        intensity_sum += df1_values[m][n]

            if positive_count > 0:
                strength = positive_count/1000
            else:
                strength = 0
            
            # 将结果写入结果矩阵
            result_matrix.at[celltype2, celltype1] = strength

# 打印结果矩阵
print(result_matrix)
result_matrix.to_csv("D:\\work\\CC_communication\\result\\CRC\\2000\\tumor\\strength.csv")
