from numpy import *
import matplotlib.pyplot as plt
import math

# 单层决策树生成函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1)) # 将返回数组的全部元素设置为1
    if threshIneq == 'lt':  # 将所有不满足不等式要求的元素设置为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0  # 用于在特征的所有可能值上进行遍历
    bestStump = {}  # 这个字典用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClassEst = mat(zeros((m, 1)))
    minError = inf  # 初始化成正无穷大
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()  # 每列最小值
        rangeMax = dataMatrix[:, i].max()  # 每列最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
            errArr = mat(ones((m, 1)))
            errArr[predictedVals == labelMat] = 0
            weightedError = D.T * errArr
            # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is % .3f" % (i, threshVal, inequal, weightedError))
            if weightedError < minError:
                minError = weightedError
                bestClassEst = predictedVals.copy()
                bestStump['dim'] = i
                bestStump['thresh'] = threshVal
                bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

# 基于单层决策树的AdaBoost训练过程
def absBoostTrainDS(dataArr, classLabels, numIt = 40): # 数据集、类别标签以及迭代次数numIt
    weakClassArr = []  # 单层决策树数组
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 被初始化成1/m
    aggClassEst = mat(zeros((m, 1)))  # 记录每个数据点的类别估计累计值
    for i in range(numIt):  # 运行numIt次或者直到训练错误率为0为止
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 返回的是具有最小错误率的单层决策树模型和最小的错误率以及估计的类别向量
        print("D：", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 该值会告诉总分类器本次单层决策树输出结果的权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels). T, classEst)  # 计算下一次迭代中的新权重向量D
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggClassEst.sum( )/m
        print("total error:", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr

# AdaBoost分类函数
def adaClassify(datToClass, classifierArr):  # 待分类样例 多个弱分类器组成的数组
    dataMatrix = mat(dataMatrix)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        # 对每个分类器得到一个类别的估计值
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)  # 程序返回aggClassEst的符号

# 自适应数据加载函数
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelsMat = []
    f = open(filename)
    for line in f.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelsMat.append(float(curLine[-1]))
    return dataMat, labelsMat

# ROC曲线的绘制及AUC计算函数
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig .clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the area under the curve is:", ySum * xStep)