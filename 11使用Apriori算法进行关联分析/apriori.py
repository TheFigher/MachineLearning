from numpy import *

# Apriori算法中的辅助函数
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
def createCl(dataset):  # 构建第一个候选项集C1
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)  # frozenset列表不可更改
def scanD(D, Ck, minSupport):  # 数据集，候选项集列表，感兴趣项集的最小支持度（此函数用于将C1生成L1）
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# Apriori算法
def aprioriGen(Lk, k):  # 频繁项集列表Lk与项集元素个数k
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:, k-2]  # 若两个集合前k-2个相等则合并
            L2 = list(Lk[j])[:, k-2]
            L1.sort() # 升序
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList
def apriori(dataSet, minSupport = 0.5):
    C1 = createCl(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k =2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)  # 创建Ck
        Lk, supK = scanD(D, Ck, minSupport)  # 基于Ck创建Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# 关联规则生成函数
def generateRules(L, supportData, minConf = 0.7):  # 频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = []
    for i in range(1, len(L)):  # 只构建两个或者更多的元素的集合
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList  # 包含可信度的规则列表
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]  # 计算可信度
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # try further merging
        Hmp1 = aprioriGen(H, m+1)  # create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #  need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

#  示例
# def pntRules(ruleList, itemMeaning):
#     for ruleTup in ruleList:
#         for item in ruleTup[0]:
#             print
#             itemMeaning[item]
#         print
#         "           -------->"
#         for item in ruleTup[1]:
#             print
#             itemMeaning[item]
#         print
#         "confidence: %f" % ruleTup[2]
#         print  # print a blank line
from time import sleep
from votesmart import votesmart

votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'


# votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = [];
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)  # api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                        (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)  # delay to be polite
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):  # this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']  # list of what each item stands for
    for billTitle in billTitleList:  # fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}  # list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning
