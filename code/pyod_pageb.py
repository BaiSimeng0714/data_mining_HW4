import pandas as pd
import numpy as np
from scipy import stats
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

for i in range(1796,1801):
    filename = "./data_pageb/pageb_benchmark_"+'{0:04}'.format(i)+".csv"
    df = pd.read_csv(filename)
    # print(df.describe())

    scaler = MinMaxScaler(feature_range=(0,1))
    # df[['V','V.1','V.2','V.3','V.4','V.5','V.6','V.7','V.8','V.9']] = scaler.fit_transform(df[['V','V.1','V.2','V.3','V.4','V.5','V.6','V.7','V.8','V.9']])
    # print(df[['V','V.1','V.2','V.3','V.4','V.5','V.6','V.7','V.8','V.9']].head())

    t1=df['ground.truth'].value_counts(normalize=True)
    t2=df['ground.truth'].value_counts(normalize=False)
    x1 = df['V'].values.reshape(-1,1)
    x2 = df['V.1'].values.reshape(-1,1)
    x3 = df['V.2'].values.reshape(-1,1)
    x4 = df['V.3'].values.reshape(-1, 1)
    x5 = df['V.4'].values.reshape(-1, 1)
    x6 = df['V.5'].values.reshape(-1, 1)
    x7 = df['V.6'].values.reshape(-1, 1)
    x8 = df['V.7'].values.reshape(-1, 1)
    x9 = df['V.8'].values.reshape(-1, 1)
    x10 = df['V.9'].values.reshape(-1, 1)
    x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10),axis=1)
    # 设置离群点数据
    random_state = np.random.RandomState(42)
    outliers_fraction = t1["anomaly"]
    outliers = t2["anomaly"]
    print("benchmark_",'{0:04}'.format(i),"的离群点共",outliers,"个，占比为",outliers_fraction,"%")
    mytxt = open('out_pageb.txt', mode='a', encoding='utf-8')
    print("benchmark_",'{0:04}'.format(i),"的离群点共", outliers, "个，占比为", outliers_fraction, "%",file=mytxt)
    mytxt.close()
    if (outliers_fraction > 0.5):
        mytxt = open('out_pageb.txt', mode='a', encoding='utf-8')
        print("离群点占比过大，放弃此benchmark", file=mytxt)
        print("\n", file=mytxt)
        mytxt.close()
        continue
    # 定义7个后续会使用的离群点检测模型
    classifiers = {
        "Angle-based Outlier Detector(ABOD)" : ABOD(contamination=outliers_fraction),
        "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination = outliers_fraction,check_estimator=False,random_state = random_state),
        "Feature Bagging" : FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state = random_state),
        "Histogram-base Outlier Detection(HBOS)" : HBOS(contamination=outliers_fraction),
        "Isolation Forest" :IForest(contamination=outliers_fraction,random_state = random_state),
        "KNN" : KNN(contamination=outliers_fraction),
        "Average KNN" :KNN(method='mean',contamination=outliers_fraction)
    }
    xx, yy, zz = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    #逐一 比较模型
    for i ,(clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(x)
        # 预测利群得分
        scores_pred = clf.decision_function(x)*-1
        # 预测数据点是否为 离群点
        y_pred = clf.predict(x)
        n_inliers = len(y_pred)-np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred==1)
        plt.figure(figsize=(10,10))

        # 复制一份数据
        dfx = df
        dfx['outlier'] = y_pred.tolist()
        # IX1 非离群点的特征1，IX2 非离群点的特征2
        IX1 = np.array(dfx['V'][dfx['outlier']==0]).reshape(-1,1)
        IX2 = np.array(dfx['V.1'][dfx['outlier']==0]).reshape(-1,1)
        IX3 = np.array(dfx['V.2'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX4 = np.array(dfx['V.3'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX5 = np.array(dfx['V.4'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX6 = np.array(dfx['V.5'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX7 = np.array(dfx['V.6'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX8 = np.array(dfx['V.7'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX9 = np.array(dfx['V.8'][dfx['outlier'] == 0]).reshape(-1, 1)
        IX10 = np.array(dfx['V.9'][dfx['outlier'] == 0]).reshape(-1, 1)
        # OX1 离群点的特征1，OX2离群点特征2
        OX1 = np.array(dfx['V'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX2 = np.array(dfx['V.1'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX3 = np.array(dfx['V.2'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX4 = np.array(dfx['V.3'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX5 = np.array(dfx['V.4'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX6 = np.array(dfx['V.5'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX7 = np.array(dfx['V.6'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX8 = np.array(dfx['V.7'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX9 = np.array(dfx['V.8'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX10 = np.array(dfx['V.9'][dfx['outlier'] == 0]).reshape(-1, 1)
        percent = n_outliers / len(df.index)
        print("模型",clf_name,"检测到的离群点有 ",n_outliers,"非离群点有",n_inliers,"离群点占比为",percent)
        mytxt = open('out_pageb.txt', mode='a', encoding='utf-8')
        print("模型",clf_name,"检测到的离群点有 ",n_outliers,"非离群点有",n_inliers,"离群点占比为",percent,file=mytxt)
        mytxt.close()

    mytxt = open('out_pageb.txt', mode='a', encoding='utf-8')
    print("\n", file=mytxt)
    mytxt.close()