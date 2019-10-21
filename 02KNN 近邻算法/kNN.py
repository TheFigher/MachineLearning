# -*- coding: utf-8 -*-
from numpy import *
import operator  # 运算符模块
from os import listdir  # 列出给定目录的文件名


# 通用函数 创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])  # 数组
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法
# 对未知类别属性的数据集中的每个点依次执行以下操作：
# (1) 计算已知类别数据集中的点与当前点之间的距离；
# (2) 按照距离递增次序排序；
# (3) 选取与当前点距离最小的k个点；
# (4) 确定前k个点所在类别的出现频率；
# (5) 返回前k个点出现频率最高的类别作为当前点的预测

def classify0(inx, dataset, labels, k): # 用于分类的输入向量是inX，输入的训练样本集为dataSet，标签向量为labels，最后的参数k表示用于选择最近邻居的数目
    dataSetSize = dataset.shape[0]  # 矩阵行数就是数据数量 shape[0]-行数 shape[1]-列数
    diffMat = tile(inx, (dataSetSize, 1)) - dataset # tile(A,reps)把数组沿着各个方向复制 m行1列
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1) # axis=0按列相加 =1按行相加
    distance = sqDistance ** 0.5  # 欧式距离公式计算距离
    sortedDistIndicies = distance.argsort()  # argsort()函数是将distance中的元素从小到大排列，提取其对应的index(索引)，然后输出到sortedDistIndicies
    classCount = {}  # 建立一个空集合
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 提取排序好了的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值（0）。
        # 按照第二个元素的次序对元组进行排序 此处的排序为逆序，即按照从最大到最小次序排序，最后返回发生频率最高的元素标签
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

# 将文本记录转换为NumPy的解析程序  输出为训练样本矩阵和类标签向量
def file2matrix(filename):
    f = open(filename)
    arrayOLines = f.readlines()
    numberOfLines = len(arrayOLines) # 得到文件行数
    returnMat = zeros((numberOfLines, 3)) # 创建一个numberOfLines行  3列的0矩阵 根据数据特征值个数确定
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 函数line.strip()截取掉所有的回车字符
        listFromLine = line.split('\t')  # 后使用tab字符\t将上一步得到的整行数据分割成一个元素列表。
        returnMat[index, :] = listFromLine[0:3]  # 选取每行前3个元素，将它们存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 将列表的最后一列存储到向量classLabelVector中
        index += 1
    return returnMat, classLabelVector

# 归一化特征值  该函数可以自动将数字特征值转化为0到1的区间 newValue = (oldValue - min)/(max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 每列的最小值
    maxVals = dataSet.max(0) # 每列的最大值
    ranges = maxVals - minVals  #  每列的极差
    normDataSet = zeros(shape(dataSet))  # 建立0矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataset = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 分类器针对网站的测试代码
def datingClassTest():
    hoRatio = 0.09 # 选取9%测试
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 9%的测试数据数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(norMat[i, :], norMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:%d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the totle error rate is: %f" % (errorCount / float(numTestVecs)))  # 错误率


# 我们将把一个32×32的二进制图像矩阵转换为1×1024的向量
def img2vector(filename):
    returnVect = zeros((1, 1024))  # 1行1024列 都为0的矩阵
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# 手写数字识别系统的测试代码
def handWritingClassTest():
    hwLabels = []  # 标签
    trainingFileList = listdir('digits/trainingDigits/')  # 打开文件夹中的文件进入列表
    m = len(trainingFileList)  # 列表长度
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits/')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is:  % d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount =+ 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    # group, labels = createDataSet()
    # print(group, labels)

    datingClassTest()
    # handWritingClassTest()