from numpy import *

# 打开文件获取数据
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# 其中i是第一个alpha的下标，m是所有alpha的数目。只要函数值不等于输入值i，函数就会进行随机选择
def selectJrand(i, m):
    j = 1
    while(j == 1):
        j = int(random.uniform(0, m))
    return j

# 是用于调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

# 简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # 数据集、类别标签、常数C、容错率和退出前最大的循环次数
    dataMatrix = mat(dataMatIn)  # mat转换成矩阵
    labelMat = mat(classLabels).transpose()  # 转置
    b = 0
    m, n = shape(dataMatrix)  # m行n列
    alphas = mat(zeros((m, 1)))  # 构建列矩阵 初始化为0
    iter = 0  # 该变量存储的则是在没有任何alpha改变的情况下遍历数据集的次数
    while (iter < maxIter):  # 当该变量达到输入值maxIter时，函数结束运行并退出
        alphaPairsChanged = 0  # 每次循环当中，将alphaPairsChanged先设为0，然后再对整个集合顺序遍历
                               # alphaPairsChanged用于记录alpha是否已经进行优化
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # multiply矩阵相乘 得到预测结果
            Ei = fXi - float(labelMat[i])  #  基于这个实例的预测结果和真实结果的比对，就可以计算误差Ei
            # 如果误差很大，那么可以对该数据实例所对应的alpha值进行优化。
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)  # 随机选择第二个alpha值
                fXj = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b  # multiply矩阵相乘 得到预测结果
                Ej = fXj - float(labelMat[j])  # 计算误差
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print("L==H")
                    continue
                #  Eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])   # update i by the same amount as j #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i] * (alphas[i]-alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j]-alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i] * (alphas[i]-alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j]-alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

