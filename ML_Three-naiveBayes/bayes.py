from numpy import *
import re
import operator
import feedparser
import warnings
warnings.simplefilter("ignore")


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1是辱骂的, 0不是
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 两个集合的并集，使用set保证词条不重复
    return list(vocabSet)  # 词汇表


# 词集模型 每个词只能才出现一次
def setOfWords2Vec(vocabList, inputSet):  # 输入词汇表与某个文档
    returnVec = [0] * len(vocabList)  # 创建一个和词汇表等长的向量，并将其元素都设置为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 输出文档向量


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):  # 输入参数为文档矩阵trainMatrix 以及由每篇文档类别标签所构成的向量trainCategory
    numTrainDocs = len(trainMatrix)  # 文档矩阵的长度
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱累概率
    p0Num = ones(numWords)  # 初始化
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denum = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 个数加一
            p1Denum += sum(trainMatrix[i])  # 总词数加一
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denum)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive  # 返回俩向量和一个概率


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):  # 要分类的向量以及trainNB0得到的三个概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  # 这属于一个二分类
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 朴素贝叶斯词袋模型 （每词可出现多次）
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 #每出现一次加一
    return returnVec


# 朴素贝叶斯过滤垃圾邮件(使用朴素贝叶斯进行交叉验证)
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)  # 接受一个大字符串并将其解析为字符串列表
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 该函数去掉少于两个字符的字符串，并将所有字符串转换为小写


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)    # 以上导入并解析文本文件
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))  # 训练集
    testSet = []   # 测试集
    for i in range(10):  # 做测试
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机取10个
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        trainMat = []
        trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # data
            trainClasses.append(classList[docIndex])                       # target
        p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
        errorCount = 0
        for docIndex in testSet:
            wordVector = setOfWords2Vec(vocabList, docList[docIndex])
            if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
        print('the error rate is:', float(errorCount) / len(testSet))


# RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]  # 出现次数从高到低排序30个

def localWords(feed1,feed0):
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

# 最具表征性的词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])



if __name__ == '__main__':
    # testingNB()
    # spamTest()
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocabList, pSF, pNY = localWords(ny, sf)
    print(getTopWords(ny,sf))