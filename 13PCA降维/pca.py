from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()] # 原理与前面打开文件的方法相似
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):  # 进行PCA操作的数据集 应用的N个特征
    meanVals = mean(dataMat, axis=0)  # 按列计算平均值
    meanRemoved = dataMat - meanVals  # 零均值化
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差 rowvar=0说明传入的数据一行代表一个样本
    eigVals,eigVects = linalg.eig(mat(covMat)) # 对协方差矩阵进行矩阵特征向量求解
    eigValInd = argsort(eigVals)            # 排序，排序从小到大
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 最大的n个特征值的下标
    redEigVects = eigVects[:,eigValInd]       # 最大的n个特征值对应的特征向量
    lowDDataMat = meanRemoved * redEigVects  # 低维特征空间的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 重构数据
    return lowDDataMat, reconMat  # 前者是pca分析后的值，后者就是主成分分析后的矩阵数据。

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])   # 计算所有非NaN的平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal          # 将所有NaN置为平均值
    return datMat


if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDMat, recoMat = pca(dataMat, 2)
    print(shape(lowDMat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(recoMat[:, 0].flatten().A[0], recoMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()
