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

for i in range(1,2):
    filename = "./data_skin/skin_benchmark_"+'{0:04}'.format(i)+".csv"
    df = pd.read_csv(filename)
    print(df.describe())
    show = False
    if show:
        plt.figure(figsize=(10,10))
        plt.scatter(df['R'],df['G'],df['B'])
        plt.xlabel("R")
        plt.ylabel("G")
        plt.zlabel("B")
        plt.show()

    scaler = MinMaxScaler(feature_range=(0,1))
    df[['R','G','B']] = scaler.fit_transform(df[['R','G','B']])
    print(df[['R','G','B']].head())
    t1=df['ground.truth'].value_counts(normalize=True)
    t2=df['ground.truth'].value_counts(normalize=False)

    x1 = df['R'].values.reshape(-1,1)
    x2 = df['G'].values.reshape(-1,1)
    x3 = df['B'].values.reshape(-1,1)
    x = np.concatenate((x1,x2,x3),axis=1)
    # 设置离群点数据
    random_state = np.random.RandomState(42)
    outliers_fraction = t1["anomaly"]
    outliers = t2["anomaly"]
    print("benchmark_",'{0:04}'.format(i),"的离群点共",outliers,"个，占比为",outliers_fraction,"%")
    mytxt = open('out_skin.txt', mode='a', encoding='utf-8')
    print("benchmark_",'{0:04}'.format(i),"的离群点共", outliers, "个，占比为", outliers_fraction, "%",file=mytxt)
    mytxt.close()
    if (outliers_fraction > 0.5):
        mytxt = open('out_skin.txt', mode='a', encoding='utf-8')
        print("离群点占比过大，放弃此benchmark", file=mytxt)
        print("\n", file=mytxt)
        mytxt.close()
        continue
    # 定义7个后续会使用的离群点检测模型
    classifiers = {
        # "Angle-based Outlier Detector(ABOD)" : ABOD(contamination=outliers_fraction),
        "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination = outliers_fraction,check_estimator=False,random_state = random_state),
        "Feature Bagging" : FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state = random_state),
        "Histogram-base Outlier Detection(HBOS)" : HBOS(contamination=outliers_fraction),
        "Isolation Forest" :IForest(contamination=outliers_fraction,random_state = random_state),
        "KNN" : KNN(contamination=outliers_fraction),
        "Average KNN" :KNN(method='mean',contamination=outliers_fraction)
    }

    #逐一 比较模型
    xx,yy,zz = np.meshgrid(np.linspace(0,1,200),np.linspace(0,1,200),np.linspace(0,1,200))
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
        IX1 = np.array(dfx['R'][dfx['outlier']==0]).reshape(-1,1)
        IX2 = np.array(dfx['G'][dfx['outlier']==0]).reshape(-1,1)
        IX3 = np.array(dfx['B'][dfx['outlier'] == 0]).reshape(-1, 1)
        # OX1 离群点的特征1，OX2离群点特征2
        OX1 = np.array(dfx['R'][dfx['outlier']==1]).reshape(-1,1)
        OX2 = np.array(dfx['G'][dfx['outlier'] == 1]).reshape(-1, 1)
        OX3 = np.array(dfx['B'][dfx['outlier'] == 1]).reshape(-1, 1)
        percent = n_outliers / len(df.index)
        print("模型",clf_name,"检测到的离群点有 ",n_outliers,"非离群点有",n_inliers,"离群点占比为",percent)
        mytxt = open('out_skin.txt', mode='a', encoding='utf-8')
        print("模型",clf_name,"检测到的离群点有 ",n_outliers,"非离群点有",n_inliers,"离群点占比为",percent,file=mytxt)
        mytxt.close()

    mytxt = open('out_skin.txt', mode='a', encoding='utf-8')
    print("\n", file=mytxt)
    mytxt.close()