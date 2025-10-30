import os
import sys

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
import time

# 开始计时
start_time = time.time()

def match_and_shuffle_columns(type1, type2):
    np.random.seed(42)
    # 获取两个矩阵的形状
    rows1, cols1 = type1.shape
    rows2, cols2 = type2.shape

    # 获取较小的行数
    min_rows = min(rows1, rows2)

    # 生成随机行索引
    type1_indices = np.arange(rows1)
    type2_indices = np.arange(rows2)
    np.random.shuffle(type1_indices)
    np.random.shuffle(type2_indices)

    # 选择随机行
    selected_type1_rows = type1_indices[:min_rows]
    selected_type2_rows = type2_indices[:min_rows]

    # 重组矩阵
    new_type1 = type1[selected_type1_rows, :]
    new_type2 = type2[selected_type2_rows, :]

    return new_type1, new_type2

def KFold_type1_and_type2_CVP(type1, type2, pcc, kfold_splits):
    relation_result = np.eye(len(type2.columns)) - np.eye(len(type2.columns))

    kfold = KFold(n_splits=kfold_splits, random_state=False, shuffle=True)

    type1 = type1.values
    type2 = type2.values

    type1,type2=match_and_shuffle_columns(type1,type2)

    for i in range(np.shape(type1)[1]):#遍历type1中genei作为因变量
        
        Y = type1[:, i]#因变量
        new_df_origin = type1 * pcc[i]
        new_df = np.delete(new_df_origin, i, 1)
        X = new_df[:, np.where(new_df.any(axis=0))[0]]

        if np.sum(X) == 0:
            continue
        for j in range(np.shape(type1)[1]):
            accumulate_casual_strength = 0
            new_df_add2 = type2[:, j] 
            new_df_add2 =  new_df_add2[:, np.newaxis]
            X2 = np.concatenate((new_df_origin, new_df_add2), axis=1)
            
            for k, (train_index, test_index) in enumerate(kfold.split(X)):
                xtrain = X[train_index]
                xtest = X[test_index]
                ytrain = Y[train_index].reshape((len(Y[train_index]), 1))
                ytest = Y[test_index].reshape((len(Y[test_index]), 1))

                xtrain2 = X2[train_index]
                xtest2 = X2[test_index]
                ytrain2 = Y[train_index].reshape((len(Y[train_index]), 1))
                ytest2 = Y[test_index].reshape((len(Y[test_index]), 1))

                reg = LR().fit(xtrain, ytrain)
                yhat = reg.predict(xtest)
                mse = MSE(yhat, ytest)
                
                reg2 = LR().fit(xtrain2, ytrain2)
                yhat2 = reg2.predict(xtest2)
                mse2 = MSE(yhat2, ytest2)

                if mse2 <= 0:
                    continue

                casual_strength = np.log(mse / mse2)
                accumulate_casual_strength = (accumulate_casual_strength + casual_strength)

            relation_result[i, j] = (accumulate_casual_strength / kfold.get_n_splits())
    return relation_result
def main():
    np.random.seed(42)
    celltype1 = sys.argv[1]
    celltype2 = sys.argv[2] 
    pcc_limit = float(sys.argv[3])
    n_splits = int(sys.argv[4])

    type1 = pd.read_csv(f"./{celltype1}/{celltype1}.csv", index_col=0).T
    type2 = pd.read_csv(f"./{celltype2}/{celltype2}.csv", index_col=0).T

    rows, cols = type1.shape
    pcc=np.corrcoef(type1.T)
    pcc=np.abs(pcc)
    pcc = pcc >= pcc_limit
    for i in range(cols):
        pcc[i,i]=0
    pcc = np.nan_to_num(pcc, nan=1)

    relation_accumulate = KFold_type1_and_type2_CVP(type1, type2, pcc, n_splits)
    relation_accumulate = pd.DataFrame(relation_accumulate)
    relation_accumulate.columns = type1.columns
    relation_accumulate.to_csv(f"./{celltype1}/{celltype1}_{celltype2}_{pcc_limit}_{n_splits}.csv")            

# 结束计时
end_time = time.time()
# 计算代码运行时间
execution_time = end_time - start_time

# 打印代码运行时间
print("Execution time:", execution_time, "seconds")

if __name__ == "__main__":
    main()
