from numpy import *
import numpy as np
import matplotlib.pyplot as plt


# 数据导入函数 划分Tab键分割的文本文件
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) -1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat

# 标准回归函数  计算最佳拟合直线
def standRegres(xArr, yArr):
    xMat = mat(xArr)  # 读入变量X
    yMat = mat(yArr).T  # 读入变量Y
    xTx = xMat.T * xMat  # 计算xTx
    if linalg.det(xTx) == 0.0:  # 判断行列式是否为0
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 局部加权线性回归函数(LWLR)
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr)  # 读取x
    yMat = mat(yArr).T  # 读取y
    m = shape(xMat)[0]  # m 行数
    weights = mat(eye((m)))  # 用来给每个数据点赋予权重（创建对角矩阵）
    for j in range(m):      # 遍历数据集
        diffMat = testPoint - xMat[j, :]  # 样本点与待预测值点之间的距离
        weights[j, j] = exp(diffMat * diffMat.T / (-2 * k ** 2))  # 随着距离的递增，权重以指数级衰减（K控制衰减速度）
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:  # 判断行列式（矩阵可逆？）
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0): # 用于为数据集中每个点调用lwlr()，这有助于求解k的大小。
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# example one
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

# 岭回归
def ridgeRegres(xArr, yArr, lam = 0.2):
    xMat = mat(xArr)
    yMat = mat(yArr)
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr): # 用于在一组λ上测试结果
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # 数据标准化  具体的做法是所有特征都减去各自的均值并除以方差
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步线性回归 # 输入数据xArr和预测变量yArr。此外还有两个参数：一个是eps，表示每次迭代需要调整的步长；另一个是numIt，表示迭代次数。
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt): # 迭代次数
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# 预测乐高玩具套装价格例子
from time import sleep
import json
import urllib.request


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

# 交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A, array(testY))
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print('the best model from ridge regression is:\n', unReg)
    print('with constant term:', -1 * sum(multiply(meanX, unReg)) + mean(yMat))















if __name__ == '__main__':
    # abX, abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX, abY)
    #
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()

    # xArr, yArr = loadDataSet('abalone.txt')
    # w = stageWise(xArr, yArr, 0.001, 5000)
    # print(w)
    #
    # xMat = mat(xArr)
    # yMat = mat(yArr).T
    # xMat = regularize(xMat)
    # yM = mean(yMat, 0)
    # yMat = yMat - yM
    # weights = standRegres(xMat, yMat.T)
    # print(weights.T)

    lgX = []
    lgY = []
    print(setDataCollect(lgX, lgY))