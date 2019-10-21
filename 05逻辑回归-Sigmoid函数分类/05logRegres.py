from numpy import *
import matplotlib.pyplot as plt

# 获取联系数据
def loadDataSet():
    dataMat = []  # 数据集
    labelMat = []  # 标签集
    fr = open('testSet.txt')  # 打开文件
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split()  # 去除首尾空格并返回
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # append()方法向列表的尾部添加一个新的元素,只接受一个参数
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# Sigmoid（阶跃）函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升算法V1.0
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) # 用mat函数转换为矩阵之后可以才进行一些线性代数的操作（实际数据没变，只是类型变化）
    labelMat = mat(classLabels).transpose()  # 转置
    m, n = shape(dataMatrix)  # 数据矩阵的大小
    alpha = 0.001   # 步长
    maxCycles = 500  # 迭代次数
    weights = ones((n, 1))  # n行1列都是1.
    for k in range(maxCycles):              # 重矩阵运算
        h = sigmoid(dataMatrix * weights)     # 矩阵多重运算  得到m行1列
        error = (labelMat - h)              # 矢量相减
        weights = weights + alpha * dataMatrix.transpose() * error  # 矩阵多重运算 误差相加
    return weights

# 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)  # 创建数组
    n = shape(dataArr)[0]  # 数组行数
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):  # 对特征数据逐行操作
        if int(labelMat[i]) == 1:  # 如果标签是1 放在1里
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:    # 否则放在2里
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升算法V1.1
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)    # 所有系数初始化为1
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法V1.2
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)   # 初始化为1
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    # alpha会随着迭代次数不断减小，但永远不会减小到0
            randIndex = int(random.uniform(0, len(dataIndex)))  # 因为常数而转到0
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  # 随机选择alpha
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# 例子（预测horse是否生病）
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:  # 如果标签结果大于0.5进入1  否则为0
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')  # 打开文件
    frTest = open('horseColicTest.txt')
    trainingSet = []  # 训练数据
    trainingLabels = []  # 训练标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):  # 共20个特征值 1个目标值
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)  # 20个特征值
        trainingLabels.append(float(currLine[21]))  # 目标值
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)  # 练习1000次
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

# 调用数colicTest()10次并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    multiTest()