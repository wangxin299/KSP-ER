import numpy as np
import pandas as pd
import operator
# from  data import get_wrong_list
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item]) #store all the item unrepeatly

    C1.sort()
    #return map(frozenset, C1)#frozen set, user can't change it.
    return list(map(frozenset, C1))

def scanD(D,Ck,minSupport):
#参数：数据集、候选项集列表 Ck以及感兴趣项集的最小支持度 minSupport
    ssCnt={}
    for tid in D:#遍历数据集
        for can in Ck:#遍历候选项
            if can.issubset(tid):#判断候选项中是否含数据集的各项
                #if not ssCnt.has_key(can): # python3 can not support
                if not can in ssCnt:
                    ssCnt[can]=1 #不含设为1
                else: ssCnt[can]+=1#有则计数加1
    numItems=float(len(D))#数据集大小
    retList = []#L1初始化
    supportData = {}#记录候选项中各个数据的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems#计算支持度
        if support >= minSupport:
            retList.insert(0,key)#满足条件加入L1中
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k): #组合，向上合并
    #creates Ck 参数：频繁项集列表 Lk 与项集元素个数 k
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): #两两组合遍历
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #若两个集合的前k-2个项相同时,则将两个集合合并
                retList.append(Lk[i] | Lk[j]) #set union
    return retList


def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet)) #python3
    L1, supportData = scanD(D, C1, minSupport)#单项最小支持度判断 0.5，生成L1
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):#创建包含更大项集的更大列表,直到下一个大的项集为空
        Ck = aprioriGen(L[k-2], k)#Ck
        Lk, supK = scanD(D, Ck, minSupport)#get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=1.0):
    #频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = [] #存储所有的关联规则
    for i in range(1, len(L)):  #只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
               # continue
            #如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:#第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)# 调用函数2
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=1.0):
    #针对项集中只有两个元素时，计算可信度
    prunedH = []#返回一个满足最小可信度要求的规则列表
    for conseq in H:#后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] #可信度计算，结合支持度数据
        if conf >= minConf:
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            #如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            brl.append((freqSet-conseq, conseq, conf))#添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)#同样需要放入列表到后面检查
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=1.0):
    #参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m+1)#存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)#计算可信度
        if (len(Hmp1) > 1):
        #满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
def lift(suppData,rules):
    dirt=[]
    for i in range(len(rules)):
        str = rules[i][1]
        supp = suppData[str]
        conf = rules[i][2]
        lif=conf/supp
        if(lif>1):
            dirt.append(rules[i])
    return dirt
def get_wrong_list():
    dirt = {}
    assist2009 = pd.read_csv('../data/assistment2009/assisment2009_apriori_input.csv', sep=',').to_numpy()
    for i in assist2009:
        user = i[0]
        kc = i[2]
        result = i[3]
        # print(user, kc, result)
        if result == 0:
            dirt.setdefault(user, []).append(kc)
    wro_list = []
    for key in dirt.keys():
        li = set(dirt[key])
        li = list(li)
        wro_list.append(li)
    # print(wro_list)
    return wro_list
if __name__ == '__main__':
    '''
    # 导入数据集
    ratings = pd.read_csv('E:/yanjiu/程序与数据/源代码/TEST/board1.csv',index_col=0)
    # change the string to float
    
    dataset = ratings.iloc[0:103,2:18]
    dataset=dataset.values.tolist()
    #去除nan
    for i in dataset:
        for j in range(len(i)-1,-1,-1):
            if np.isnan(i[j]):
                del i[j]
    #print(dataset)
    #myDat = loadDataSet()
    # 选择频繁项集
    '''
    dataset=get_wrong_list()
    L, suppData = apriori(dataset , 0.05)
    print("频繁项集L：", L)
    print("所有候选项集的支持度信息：", suppData)

    rules = generateRules(L, suppData, minConf=0.7)

    data = open("../result/assistment2009/assist2009_apriori.txt", 'w')
    print(rules, file=data)
    data.close()

    # for i in rules:
    #     print(i[0],' -> ',i[1],' = ',float(1.0/dic[i[0]]))
    print('关联规则: ', rules)
    s=set()
    for i in rules:
        s.add(i[0])
    for i in s:
        print(i,':',end='')
        for j in rules:
            if j[0]==i:
                print(j[1],',',end='')
        print()



