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
def KFold_CVP(type1, pc_limit, kfold_splits):
    relation_result = np.eye(len(type1.columns)) - np.eye(len(type1.columns))

    pcc=np.corrcoef(type1.T)
    pcc=np.abs(pcc)
    pcc = pcc >= pc_limit
    for i in range(np.shape(type1)[1]):
        pcc[i,i]=0
    pcc = np.nan_to_num(pcc, nan=1)

    kfold = KFold(n_splits=kfold_splits, random_state=False, shuffle=True)

    type1 = type1.values

    for i in range(np.shape(type1)[1]):#遍历type1中genei作为因变量
        Y = type1[:, i]
        df = type1 * pcc[i]
        X = np.delete(df, i, 1)
        
        new_X = X[:, np.where(X.any(axis=0))[0]]
        if np.sum(new_X) == 0:
            continue
        for j, element in enumerate(np.where(X.any(axis=0))[0]):
            accumulate_casual_strength = 0
            X2 = np.delete(X, element, 1)
            new_X2 = X2[:, np.where(X2.any(axis=0))[0]]
            if np.sum(new_X2) == 0:
                continue
            for k, (train_index, test_index) in enumerate(kfold.split(X)):
                xtrain = new_X[train_index]
                xtest = new_X[test_index]
                ytrain = Y[train_index].reshape((len(Y[train_index]), 1))
                ytest = Y[test_index].reshape((len(Y[test_index]), 1))

                xtrain2 = new_X2[train_index]
                xtest2 = new_X2[test_index]
                ytrain2 = Y[train_index].reshape((len(Y[train_index]), 1))
                ytest2 = Y[test_index].reshape((len(Y[test_index]), 1))

                reg = LR().fit(xtrain, ytrain)
                yhat = reg.predict(xtest)
                mse = MSE(yhat, ytest)

                if mse <= 0:
                    continue

                reg2 = LR().fit(xtrain2, ytrain2)
                yhat2 = reg2.predict(xtest2)
                mse2 = MSE(yhat2, ytest2)

                casual_strength = np.log(mse2 / mse)
                accumulate_casual_strength = (accumulate_casual_strength + casual_strength)

            if i <= element:
                relation_result[i, element + 1] = accumulate_casual_strength / kfold.get_n_splits()
            else:
                relation_result[i, element] = accumulate_casual_strength / kfold.get_n_splits()
    return relation_result

def main():
    np.random.seed(42)
    celltype1 = sys.argv[1]
    celltype2 = sys.argv[2] 
    pcc_limit = float(sys.argv[3])
    n_splits = int(sys.argv[4])

    type1 = pd.read_csv(f"./{celltype1}/{celltype1}.csv", index_col=0).T
    type2 = pd.read_csv(f"./{celltype2}/{celltype2}.csv", index_col=0).T

    relation_accumulate = KFold_CVP(type1, pcc_limit, n_splits)
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
