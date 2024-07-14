'''
import sys
import os
import logging


# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, 'print.txt')

# Configure the logger
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(message)s')
# Print Python version
logging.info(f"Python version: {sys.version}\n")
logging.info(f"Python executable: {sys.executable}\n")
'''
import random
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import Counter
from itertools import combinations
import os
from collections import defaultdict

wd = os.path.dirname(__file__)

class Network_Pcard_V2_2_BN_dropout(nn.Module): # this network considers public cards of landlord
                                 # predict opponent states (as frequency of cards) (UPPER, LOWER)
                                 # does not read in action
                                 # output lstm encoding
    def __init__(self, z, nh=5, y=1, x=15, lstmsize=512, hiddensize=512, dropout_rate=0.5):
        super(Network_Pcard_V2_2_BN_dropout, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = nh # self, unavail, card count, visible, ROLE, NO action
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        self.dropout = nn.Dropout(dropout_rate)

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstmsize, batch_first=True)

        # Calculate the size of the non-historical input
        input_size = self.non_hist * y * x + lstmsize  # +lstmsize for the LSTM output
        hidden_size = hiddensize

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 2*x) # output is 2 states (Nelems)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Extract the historical layers for LSTM processing
        historical_data = x[:, self.non_hist:, :, :]
        historical_data = historical_data.flip(dims=[1])
        historical_data = historical_data.reshape(-1, self.nhist, self.y * self.x)  # Reshape for LSTM

        # Process historical layers with LSTM
        lstm_out, _ = self.lstm(historical_data)
        lstm_out = lstm_out[:, -1, :]  # Use only the last output of the LSTM

        # Extract and flatten the non-historical part of the input
        non_historical_data = x[:, :self.non_hist, :, :]
        non_historical_data = non_historical_data.reshape(-1, non_historical_data.shape[1] * self.y * self.x)

        # Concatenate LSTM output with non-historical data
        x = torch.cat((lstm_out, non_historical_data), dim=1)

        # Process through FNN as before
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x1 = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = x + x1
        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = x + x1
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))*4 # max count is 4, min count is 0
        return x, lstm_out


class Network_Qv_Universal_V1_2_BN_dropout(nn.Module): # this network uses estimated state to estimate q values of action
                               # use 3 states (SELF, UPPER, LOWER), 1 role, and 1 action (Nelems) (z=5)
                               # cound use more features such as history
                               # should be simpler
                               # use lstm and Bigstate altogether

    def __init__(self, input_size, lstmsize, hsize=256, dropout_rate=0.5):
        super(Network_Qv_Universal_V1_2_BN_dropout, self).__init__()

        hidden_size = hsize

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(input_size + lstmsize, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1) # output is q values
        self.flatten = nn.Flatten()

    def forward(self, x):

        # Process through FNN
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = x + x1

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc3(x))
        x = x + x1

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc4(x))
        x = x + x1

        x = torch.sigmoid(self.fc5(x))
        return x


SLM = Network_Pcard_V2_2_BN_dropout(15+7, 7, y=1, x=15, lstmsize=256, hiddensize=512)
QV = Network_Qv_Universal_V1_2_BN_dropout(11*15,256,512)

SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_H15-V2_3.0_0100000000.pt')))
QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_H15-V2_3.0_0100000000.pt')))
SLM.eval()
QV.eval()

# 红桃 方块 黑桃 草花
# 3 4 5 6 7 8 9 10 J Q K A 2 joker & Joker
# (0-h3 1-d3 2-s3 3-c3) (4-h4 5-d4 6-s4 7-c4) …… 52-小王->16 53-大王->17

full_input = json.loads(input())
my_history = full_input["responses"]
use_info = full_input["requests"][0]
poker, history, publiccard = use_info["own"], use_info["history"], use_info["publiccard"]
last_history = full_input["requests"][-1]["history"]
currBotID = 0 # 判断自己是什么身份，地主0 or 农民甲1 or 农民乙2
if len(history[0]) == 0:
    if len(history[1]) != 0:
        currBotID = 1
else:
    currBotID = 2
history = history[2-currBotID:]

for i in range(len(my_history)):
    history += [my_history[i]]
    history += full_input["requests"][i+1]["history"]

lenHistory = len(history)

for tmp in my_history:
    for j in tmp:
        poker.remove(j)
poker.sort() # 用0-53编号的牌

def ordinalTransfer(poker):
    newPoker = [int(i/4)+3 for i in poker if i <= 52]
    if 53 in poker:
        newPoker += [17]
    return newPoker

def transferOrdinal(subPoker, newPoker, poker):
    singlePoker, res = list(set(subPoker)), []
    singlePoker.sort()
    for i in range(len(singlePoker)):
        tmp = singlePoker[i]
        idx = newPoker.index(tmp)
        num = subPoker.count(tmp)
        res += [poker[idx + i] for i in range(num)]
    return res

def separate(poker): # 拆分手牌牌型并组成基本牌集合，返回的只是点数
    res = []
    if len(poker) == 0:
        return res
    myPoker = [i for i in poker]
    newPoker = ordinalTransfer(myPoker)
    if 16 in newPoker and 17 in newPoker: # 单独列出火箭
        newPoker = newPoker[:-2]
        res += [[16, 17]]
    elif 16 in newPoker:
        newPoker = newPoker[:-1]    
        res += [[16]]
    elif 17 in newPoker:
        newPoker = newPoker[:-1] 
        res += [[17]]
        
    singlePoker = list(set(newPoker)) # 都有哪些牌
    singlePoker.sort()

    for i in singlePoker:    # 分出炸弹，其实也可以不分，优化点之一
        if newPoker.count(i) == 4:
            idx = newPoker.index(i)
            res += [ newPoker[idx:idx+4] ]
            newPoker = newPoker[0:idx] + newPoker[idx+4:]

    # 为了简便处理带2的情形，先把2单独提出来
    specialCount, specialRes = 0, []
    if 15 in newPoker:
        specialCount = newPoker.count(15)
        idx = newPoker.index(15)
        specialRes = [15 for i in range(specialCount)]
        newPoker = newPoker[:-specialCount]

    def findSeq(p, dupTime, minLen): # 这里的p是点数，找最长的顺子，返回值为牌型组合
        resSeq, tmpSeq = [], []
        singleP = list(set(p))
        singleP.sort()
        for curr in singleP:
            if p.count(curr) >= dupTime:
                if len(tmpSeq) == 0:
                    tmpSeq = [curr]
                    continue
                elif curr == (tmpSeq[-1] + 1):
                    tmpSeq += [curr]
                    continue
            if len(tmpSeq) >= minLen:
                tmpSeq = [i for i in tmpSeq for j in range(dupTime)]
                resSeq += [tmpSeq]
            tmpSeq = []
        return resSeq

    def subSeq(p, subp): # 一定保证subp是p的子集
        singleP = list(set(subp))
        singleP.sort()
        for curr in singleP:
            idx = p.index(curr)
            countP = subp.count(curr)
            p = p[0:idx] + p[idx+countP:]
        return p
    
    # 单顺：1，5；双顺：2，3；飞机：3，2；航天飞机：4，2。因为前面已经把炸弹全都提取出来，所以这里就不主动出航天飞机了

    para = [[1,5],[2,3],[3,2]]
    validChoice = [0,1,2]
    allSeq = [[], [], []] # 分别表示单顺、双顺、三顺（飞机不带翼）
    restRes = []
    while(True): # myPoker，这里会找完所有的最长顺子
        if len(newPoker) == 0 or len(validChoice) == 0:
            break
        dupTime = random.choice(validChoice)
        tmp = para[dupTime]
        newSeq = findSeq(newPoker, tmp[0], tmp[1])
        for tmpSeq in newSeq:
            newPoker = subSeq(newPoker, tmpSeq)
        if len(newSeq) == 0:
            validChoice.remove(dupTime)
        else:
            allSeq[dupTime] += [tmpSeq]
    res += allSeq[0] + allSeq[1] # 对于单顺和双顺没必要去改变
    plane = allSeq[2]

    allRetail = [[], [], []] # 分别表示单张，对子，三张
    singlePoker = list(set(newPoker)) # 更新目前为止剩下的牌，newPoker和myPoker是一一对应的
    singlePoker.sort()
    for curr in singlePoker:
        countP = newPoker.count(curr)
        allRetail[countP-1] += [[curr for i in range(countP)]]

    # 接下来整合有需要的飞机or三张 <-> 单张、对子。这时候的飞机和三张一定不会和单张、对子有重复。
    # 如果和单张有重复，即为炸弹，而这一步已经在前面检测炸弹时被检测出
    # 如果和对子有重复，则同一点数的牌有5张，超出了4张

    # 先整合飞机
    for curr in plane:
        lenKind = int(len(curr) / 3)
        tmp = curr
        for t in range(2): # 分别试探单张和对子的个数是否足够
            tmpP = allRetail[t]
            if len(tmpP) >= lenKind:
                tmp += [i[j] for i in tmpP[0:lenKind] for j in range(t+1)]
                allRetail[t] = allRetail[t][lenKind:]
                break
        res += [tmp]

    if specialCount == 3:
        allRetail[2] += [specialRes]
    elif specialCount > 0 and specialCount <= 2:
        allRetail[specialCount - 1] += [specialRes]
    # 之后整合三张
    for curr in allRetail[2]: # curr = [1,1,1]
        tmp = curr
        for t in range(2):
            tmpP = allRetail[t]
            if len(tmpP) >= 1:
                tmp += tmpP[0]
                allRetail[t] = allRetail[t][1:]
                break
        res += [tmp]
    
    res += allRetail[0] + allRetail[1]
    return res

def checkPokerType(poker): # poker：list，表示一个人出牌的牌型
    poker.sort()
    lenPoker = len(poker)
    newPoker = ordinalTransfer(poker)
    # J,Q,K,A,2-11,12,13,14,15
    # 单张：1 一对：2 三带：零3、一4、二5 单顺：>=5 双顺：>=6
    # 四带二：6、8 飞机：>=6
    typeP, mP, sP = "空", newPoker, []

    for tmp in range(2):
        if tmp == 1:
            return "错误", poker, [] # 没有判断出任何牌型，出错
        if lenPoker == 0: # 没有牌，也即pass
            break
        if poker == [52, 53]:
            typeP = "火箭"
            break
        if lenPoker == 4 and newPoker.count(newPoker[0]) == 4:
            typeP = "炸弹"
            break
        if lenPoker == 1:
            typeP = "单张"
            break
        if lenPoker == 2:
            if newPoker.count(newPoker[0]) == 2:
                typeP = "一对"
                break
            continue
                
        firstPoker = newPoker[0]

        # 判断是否是单顺
        if lenPoker >= 5 and 15 not in newPoker:
            singleSeq = [firstPoker+i for i in range(lenPoker)]
            if newPoker == singleSeq:
                typeP = "单顺"
                break

        # 判断是否是双顺
        if lenPoker >= 6 and lenPoker % 2 == 0 and 15 not in newPoker:
            pairSeq = [firstPoker+i for i in range(int(lenPoker / 2))]
            pairSeq = [j for j in pairSeq for i in range(2)]
            if newPoker == pairSeq:
                typeP = "双顺"
                break

        thirdPoker = newPoker[2]
        # 判断是否是三带
        if lenPoker <= 5 and newPoker.count(thirdPoker) == 3:
            mP, sP = [thirdPoker for k in range(3)], [k for k in newPoker if k != thirdPoker]
            if lenPoker == 3:
                typeP = "三带零"
                break
            if lenPoker == 4:
                typeP = "三带一"
                break
            if lenPoker == 5:
                typeP = "三带二"
                if sP[0] == sP[1]:
                    break
                continue

        if lenPoker < 6:
            continue

        fifthPoker = newPoker[4]
        # 判断是否是四带二
        if lenPoker == 6 and newPoker.count(thirdPoker) == 4:
            typeP, mP = "四带两只", [thirdPoker for k in range(4)]
            sP = [k for k in newPoker if k != thirdPoker]
            if sP[0] != sP[1]:
                break
            continue
        if lenPoker == 8:
            typeP = "四带两对"
            mP, sP = [], []
            if newPoker.count(thirdPoker) == 4:
                mP, sP = [thirdPoker for k in range(4)], [k for k in newPoker if k != thirdPoker]
            elif newPoker.count(fifthPoker) == 4:
                mP, sP = [fifthPoker for k in range(4)], [k for k in newPoker if k != fifthPoker]
            if len(sP) == 4:
                if sP[0] == sP[1] and sP[2] == sP[3] and sP[0] != sP[2]:
                    break

        # 判断是否是飞机or航天飞机
        singlePoker = list(set(newPoker)) # 表示newPoker中有哪些牌种
        singlePoker.sort()
        mP, sP = newPoker, []
        dupTime = [newPoker.count(i) for i in singlePoker] # 表示newPoker中每种牌各有几张
        singleDupTime = list(set(dupTime)) # 表示以上牌数的种类
        singleDupTime.sort()

        if len(singleDupTime) == 1 and 15 not in singlePoker: # 不带翼
            lenSinglePoker, firstSP = len(singlePoker), singlePoker[0]
            tmpSinglePoker = [firstSP+i for i in range(lenSinglePoker)]
            if singlePoker == tmpSinglePoker:
                if singleDupTime == [3]: # 飞机不带翼
                    typeP = "飞机不带翼"
                    break
                if singleDupTime == [4]: # 航天飞机不带翼
                    typeP = "航天飞机不带翼"
                    break

        def takeApartPoker(singleP, newP):
            m = [i for i in singleP if newP.count(i) >= 3]
            s = [i for i in singleP if newP.count(i) < 3]
            return m, s

        m, s = [], []
        if len(singleDupTime) == 2 and singleDupTime[0] < 3 and singleDupTime[1] >= 3:
            c1, c2 = dupTime.count(singleDupTime[0]), dupTime.count(singleDupTime[1])
            if c1 != c2 and not (c1 == 4 and c2 == 2): # 带牌的种类数不匹配
                continue
            m, s = takeApartPoker(singlePoker, newPoker) # 都是有序的
            if 15 in m:
                continue
            lenm, firstSP = len(m), m[0]
            tmpm = [firstSP+i for i in range(lenm)]
            if m == tmpm: # [j for j in pairSeq for i in range(2)]
                m = [j for j in m for i in range(singleDupTime[1])]
                s = [j for j in s for i in range(singleDupTime[0])]
                if singleDupTime[1] == 3:
                    if singleDupTime[0] == 1:
                        typeP = "飞机带小翼"
                        mP, sP = m, s
                        break
                    if singleDupTime[0] == 2:
                        typeP = "飞机带大翼"
                        mP, sP = m, s
                        break
                elif singleDupTime[1] == 4:
                    if singleDupTime[0] == 1:
                        typeP = "航天飞机带小翼"
                        mP, sP = m, s
                        break
                    if singleDupTime[0] == 2:
                        typeP = "航天飞机带大翼"
                        mP, sP = m, s
                        break
        
    omP, osP = [], []
    for i in poker:
        tmp = int(i/4)+3
        if i == 53:
            tmp = 17
        if tmp in mP:
            omP += [i]
        elif tmp in sP:
            osP += [i]
        else:
            return "错误", poker, []
    return typeP, omP, osP
def recover(h): # 只考虑倒数3个，返回最后一个有效牌型及主从牌，且返回之前有几个人选择了pass；id是为了防止某一出牌人在某一牌局后又pass，然后造成连续pass
    typeP, mP, sP, countPass = "空", [], [], 0
    for i in range(-1,-3,-1):
        lastPoker = h[i]
        typeP, mP, sP = checkPokerType(lastPoker)
        if typeP == "空":
            countPass += 1
            continue
        break
    return typeP, mP, sP, countPass
def searchCard(poker, objType, objMP, objSP): # 搜索自己有没有大过这些牌的牌
    if objType == "火箭": # 火箭是最大的牌
        return []
    # poker.sort() # 要求poker是有序的，使得newPoker一般也是有序的
    newPoker = ordinalTransfer(poker)
    singlePoker = list(set(newPoker)) # 都有哪些牌
    singlePoker.sort()
    countPoker = [newPoker.count(i) for i in singlePoker] # 这些牌都有几张
    
    res = []
    idx = [[i for i in range(len(countPoker)) if countPoker[i] == k] for k in range(5)] # 分别有1,2,3,4的牌在singlePoker中的下标
    quadPoker = [singlePoker[i] for i in idx[4]]
    flag = 0
    if len(poker) >= 2:
        if poker[-2] == 52 and poker[-1] == 53:
            flag = 1

    if objType == "炸弹":
        for curr in quadPoker:
            if curr > newObjMP[0]:
                res += [[(curr-3)*4+j for j in range(4)]]
        if flag:
            res += [[52,53]]
        return res

    newObjMP, lenObjMP = ordinalTransfer(objMP), len(objMP)
    singleObjMP = list(set(newObjMP)) # singleObjMP为超过一张的牌的点数
    singleObjMP.sort()
    countObjMP, maxObjMP = newObjMP.count(singleObjMP[0]), singleObjMP[-1]
    # countObjMP虽取首元素在ObjMP中的个数，但所有牌count应相同；countObjMP * len(singleObjMP) == lenObjMP

    newObjSP, lenObjSP = ordinalTransfer(objSP), len(objSP) # 只算点数的对方拥有的主牌; 对方拥有的主牌数
    singleObjSP = list(set(newObjSP)) 
    singleObjSP.sort()
    countObjSP = 0
    if len(objSP) > 0: # 有可能没有从牌，从牌的可能性为单张或双张
        countObjSP = newObjSP.count(singleObjSP[0])

    tmpMP, tmpSP = [], []

    for j in range(1, 16 - maxObjMP):
        tmpMP, tmpSP = [i + j for i in singleObjMP], []
        if all([newPoker.count(i) >= countObjMP for i in tmpMP]): # 找到一个匹配的更大解
            if j == (15 - maxObjMP) and countObjMP != lenObjMP: # 与顺子有关，则解中不能出现2（15）
                break
            if lenObjSP != 0:
                tmpSP = list(set(singlePoker)-set(tmpMP))
                tmpSP.sort()
                tmpSP = [i for i in tmpSP if newPoker.count(i) >= countObjSP] # 作为从牌有很多组合方式，是优化点
                species = int(lenObjSP/countObjSP)
                if len(tmpSP) < species: # 剩余符合从牌特征的牌种数少于目标要求的牌种数，比如334455->lenObjSP=6,countObjSP=2,tmpSP = [8,9]
                    continue
                tmp = [i for i in tmpSP if newPoker.count(i) == countObjSP]
                if len(tmp) >= species: # 剩余符合从牌特征的牌种数少于目标要求的牌种数，比如334455->lenObjSP=6,countObjSP=2,tmpSP = [8,9]
                    tmpSP = tmp
                tmpSP = tmpSP[0:species]
            tmpRes = []
            idxMP = [newPoker.index(i) for i in tmpMP]
            idxMP = [i+j for i in idxMP for j in range(countObjMP)]
            idxSP = [newPoker.index(i) for i in tmpSP]
            idxSP = [i+j for i in idxSP for j in range(countObjSP)]
            idxAll = idxMP + idxSP
            tmpRes = [poker[i] for i in idxAll]
            res += [tmpRes]
    
    if objType == "单张": # 以上情况少了上家出2，本家可出大小王的情况
        if 52 in poker and objMP[0] < 52:
            res += [[52]]
        if 53 in poker:
            res += [[53]]

    for curr in quadPoker: # 把所有炸弹先放进返回解
        res += [[(curr-3)*4+j for j in range(4)]]
    if flag:
        res += [[52,53]]
    return res

lastTypeP, lastMP, lastSP, countPass = recover(last_history)

def n2t(n): # convert card to tensor index
    if n < 52:
        return n // 4
    else:
        return n - 52 + 13



#f = open(os.path.join('logs.txt'),'a')

import logging
# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, 'print.txt')

# Configure the logger
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(message)s')


#f.write(f'\n\n\nID: {currBotID}\n')

player = torch.zeros((15),dtype=torch.int8)
for n in poker:
    player[n2t(n)] += 1
#f.write(f'My card  {player}\n')

visible = torch.zeros((15),dtype=torch.int8)
for n in publiccard:
    visible[n2t(n)] += 1
#f.write(f'Public card  {visible}\n')

torch_hist = torch.zeros((15,15),dtype=torch.int8)
recenthist = history[-15:]
for _ in range(len(recenthist)):
    torch_hist = torch.roll(torch_hist,1,dims=0)
    for n in recenthist[_]:
        torch_hist[0][n2t(n)] += 1
#f.write(f'Tensor history  \n{torch_hist}\n\n')

played_cards = torch.zeros((3,15),dtype=torch.int8)
for _ in range(len(history)):
    pturn = _ % 3
    for n in history[_]:
        played_cards[pturn][n2t(n)] += 1

#f.write(f'Played cards  \n{played_cards}\n\n')

# lastmove is the last non zero move
# forcemove is whether last two moves are all zero
laststate = torch.zeros((15),dtype=torch.int8)

for i, rh in enumerate(recenthist[-3:][::-1]):
    if len(rh)>0:
        for n in rh:
            laststate[n2t(n)] += 1
        break

Forcemove = False or len(poker) == 20
if torch_hist[:2].sum() == 0:
    Forcemove = True

if Forcemove:
    laststate = torch.zeros((15),dtype=torch.int8)

#f.write(f'Laststate  \n{laststate}\n')
#f.write(f'Forcemove:  {Forcemove}\n')
#f.close()



r2c_base = {0:'3',
       1:'4',
       2:'5',
       3:'6',
       4:'7',
       5:'8',
       6:'9',
       7:'X',
       8:'J',
       9:'Q',
       10:'K',
       11:'A',
       12:'2',
       13:'B',
       14:'R'}

r2c_base_arr = np.array(list(r2c_base.values()))
c2r_base = {r2c_base[k]:k for k in r2c_base.keys()}

def state2list_1D(Nelems):
    # Ensure Nelems is a numpy array and r2c_base is appropriately sized
    #Nelems = np.asarray(Nelems)
    #assert len(Nelems) == 15
    #assert len(r2c_base) == 15
    
    # Use numpy.repeat to replicate r2c_base elements according to Nelems
    result = np.repeat(r2c_base_arr, Nelems.astype(int))
    
    return result.tolist()

def all_bombs_new(Nelems):
    quad = r2c_base_arr[Nelems==4]
    out = [[Q*4,(4,c2r_base[Q[0]])] for Q in quad]
    if Nelems[-2:].sum() == 2:
        out.append(['BR',(4,13)])
    return out

def all_actions(mystate,forcemove=0): # now mystate is the same shape as Nelems
    #mystate = str2state('567899XXXJQKKKB')
    # opinfo is (type, rank)
    # action is string of stuff '34567' or 'BR' for black.red joker
    # return as "action, type, rank"
    
    # return all

    # 0 pass
    # Nelems = mystate.sum(dim=0).numpy()
    Nelems = mystate.numpy()

    if not forcemove:
        out = [['',(0,0)]]
    else:
        out = []

    # 1 single
    idx_S = np.arange(15)[Nelems>0]
    sin = r2c_base_arr[idx_S]
    LenS = len(sin)
    out.extend([[S,(1,c2r_base[S])] for S in sin])

    # 2 double
    idx_D = np.arange(15)[Nelems>=2]
    dbl = r2c_base_arr[idx_D]
    LenD = len(dbl)
    out.extend([[D*2,(2,c2r_base[D[0]])] for D in dbl])

    # 3 triple
    tri = r2c_base_arr[Nelems>=3]
    out.extend([[T*3,(3,c2r_base[T[0]])] for T in tri])

    # 4 bomb
    quad = r2c_base_arr[Nelems==4]
    out.extend([[Q*4,(4,c2r_base[Q[0]])] for Q in quad])
    if Nelems[-2:].sum() == 2:
        out += [['BR',(4,13)]]

    # 5 3+1
    out.extend([[T*3 + S,(5,c2r_base[T[0]])] for T in tri for S in sin if T != S])

    # 6 3+2
    out.extend([[T*3 + D*2,(6,c2r_base[T[0]])] for T in tri for D in dbl if T != D]) # use existing tri

    # 7 4+1+1
    #quad = [[r2c_base[i]*4,(7,i)] for i in range(15) if Nelems[i] >= 4]
    #quad = r2c_base_arr[Nelems==4]
    #sin1 = [r2c_base[i]*1 for i in range(15) if Nelems[i] >= 1]
    for i in range(LenS):
        for j in range(i+1):
            if i != j or Nelems[idx_S[i]] >= 2:#mystate[:,c2r_base[s1]].sum() >= 2:
                s1,s2 = sin[i], sin[j]
                #print(i,j,s1,s2)
                out.extend([[Q*4+s1+s2,(7,c2r_base[Q])] for Q in quad if Q != s1 if Q != s2])
    
    # 8 4+2+2
    #dbl1 = [r2c_base[i]*2 for i in range(15) if mystate[:,i].sum() >= 2]
    #for d1 in dbl1:
    #    for d2 in dbl1:
    for i in range(LenD):
        for j in range(i+1):
            if i != j or Nelems[idx_D[i]] >= 4:
                d1,d2 = dbl[i], dbl[j]
                #print(d1,d2,i,j,Nelems,Nelems[idx_D[i]])
                out.extend([[Q*4+d1*2+d2*2,(8,c2r_base[Q])] for Q in quad if Q != d1 if Q != d2])

    #print(out)
    #quit()
    # 9 sequence
    # for all element find seq
    i = 0
    while i < 8:
    #for i in range(0,8):
        l = 5
        if 0 not in Nelems[i:i+l]:
            out.append([''.join(r2c_base_arr[i:i+l]),(9,i)])
            #print(i,l,Nelems[l],i+l)
            while Nelems[i+l] != 0 and i+l<12:
                l += 1
                out.append([''.join(r2c_base_arr[i:i+l]),(9,i)])
        i += 1
        #else:
        #    i += 1#np.argmin(Nelems[i:i+l])+1
    #print(out)
    #quit()

    # 10 double seq
    #i = 0
    for i in range(0,10):
    #while i < 10:
        l = 3
        if Nelems[i:i+l].min()>=2:
            out.append([''.join([D*2 for D in r2c_base_arr[i:i+l]]),(10,i)])
            #print(i,l,Nelems[l],i+l)
            while Nelems[i+l] >= 2 and i+l<12:
                l += 1
                out.append([''.join([D*2 for D in r2c_base_arr[i:i+l]]),(10,i)])
        #    i += 1
        #else:
        #    i += np.argmin(Nelems[i:i+l])+1

    # 11 triple seq
    triseq = []
    #i = 0
    for i in range(0,11):
    #while i < 11:
        l = 2
        if Nelems[i:i+l].min()>=3:
            triseq.append([''.join([T*3 for T in r2c_base_arr[i:i+l]]),(11,i)])
            #print(i,l,Nelems[l],i+l)
            while Nelems[i+l] >= 3 and i+l<12:
                l += 1
                triseq.append([''.join([T*3 for T in r2c_base_arr[i:i+l]]),(11,i)])
        #    i += 1
        #else:
        #    i += np.argmin(Nelems[i:i+l])+1

    out += triseq
    #print(out)
    #quit()
    stls = state2list_1D(Nelems)
    
    for ts in triseq:
        l = int(len(ts[0])/3)
        if l < 6: # 12 plane + wings of size 1
            tsls = [t for t in ts[0]]
            counterA = Counter(tsls)
            counterB = Counter(stls)
            others = list((counterB - counterA).elements())
            combinations_string = list(set(combinations(others, l)))
            for comb in combinations_string:
                out.append([ts[0]+''.join(comb),(12,ts[1][1])])
        if l < 5: # 13 plane + wings of size 2
            tsls = [t for t in ts[0]]
            counterA = Counter(tsls)
            counterB = Counter(stls)
            others = list((counterB - counterA).elements())
            counterC = Counter(others)
            comb = []
            for k in list(counterC.keys()):
                if counterC[k] >= 2:
                    comb.append(k*2)
                    if counterC[k] == 4:
                        comb.append(k*2)
            combinations_string = list(set(combinations(comb, l)))
            for comb in combinations_string:
                out.append([ts[0]+''.join(comb),(13,ts[1][1])])
    #print('a')
    #print(out)
    #quit()
    return out

def avail_actions(opact,opinfo,mystate,forcemove=0): # now mystate is the same shape as Nelems
    # opinfo is (type, rank)
    # action is string of stuff '34567' or 'BR' for black.red joker
    # return as "action, type, rank"
    #print(opinfo)
    if opinfo[0] != 0:

        # Nelems = mystate.sum(dim=0).numpy()
        Nelems = mystate.numpy()

        # Find action based on previous state
        # Assign unique type to new actions
        out = []
        if opinfo[0] == 1: # single
            # return all larger single or bombs
            out = [[r2c_base[i],(1,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 1]
            #sin = r2c_base_arr[Nelems>0]
            #out = [[S,(1,c2r_base[S])] for S in sin if c2r_base[S] > opinfo[1]]
            out.extend(all_bombs_new(Nelems))
            #out += all_bombs(mystate)
            #return out
        
        elif opinfo[0] == 2: # double
            # return all larger double or bombs
            out = [[r2c_base[i]*2,(2,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 2]
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 3: # triple
            # return all larger double or bombs
            out = [[r2c_base[i]*3,(3,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 3]
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 4: # bomb
            # return all larger bombs
            out = [[r2c_base[i]*4,(4,i)] for i in range(opinfo[1]+1,15) if Nelems[i] == 4]

            if Nelems[-2:].sum() == 2:
                #return out + [['BR',(4,13)]]
                out.append(['BR',(4,13)])
            else:
                #return out
                pass
        
        elif opinfo[0] == 5: # 3+1
            # return all larger trio + any 1 and bombs
            tri = [[r2c_base[i]*3,(5,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 3]
            #sin = [r2c_base[i] for i in range(15) if mystate[-1][i] >= 1]
            sin = r2c_base_arr[Nelems>0]
            out = [[t[0] + s,t[1]] for t in tri for s in sin if t[0][0] != s]
            out.extend(all_bombs_new(Nelems))
            #return out

        elif opinfo[0] == 6: # 3+2
            # return all larger trio + any 2 and bombs
            tri = [[r2c_base[i]*3,(6,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 3]
            dbl = r2c_base_arr[Nelems>1]
            #dbl = [r2c_base[i]*2 for i in range(15) if mystate[:,i].sum() >= 2]
            out = [[t[0] + d*2,t[1]] for t in tri for d in dbl if t[0][0] != d]
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 7: # 4 + 1 + 1
            # return all larger trio + any 1+1 and bombs
            quad = [[r2c_base[i]*4,(7,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 4]
            sin1 = r2c_base_arr[Nelems>0]
            out = []
            for s1 in sin1:
                for s2 in sin1:
                    if c2r_base[s1] < c2r_base[s2] or (c2r_base[s1] == c2r_base[s2] and Nelems[c2r_base[s1]] >= 2):
                        out += [[t[0]+s1+s2,t[1]] for t in quad if t[0][0] != s1 if t[0][0] != s2]
            out.extend(all_bombs_new(Nelems))
            #return out

        elif opinfo[0] == 8: # 4 + 2 + 2
            # return all larger trio + any other 1+1 and bombs
            quad = [[r2c_base[i]*4,(8,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 4]
            dbl1 = r2c_base_arr[Nelems>1]
            out = []
            for d1 in dbl1:
                for d2 in dbl1:
                    if c2r_base[d1] < c2r_base[d2] or (c2r_base[d1] == c2r_base[d2] and Nelems[c2r_base[d1]] >= 4):
                        out += [[t[0]+d1*2+d2*2,t[1]] for t in quad if t[0][0] != d1 if t[0][0] != d2]
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 9: # abcde sequence
            # return all larger sequences of same length
            l = len(opact)
            out = [[''.join(r2c_base[j] for j in range(i,i+l)),(9,i)] for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 1]
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 10: # xxyyzz double sequence
            # return all larger d sequences of same length
            l = int(len(opact)/2)
            out = [[''.join(r2c_base[j]*2 for j in range(i,i+l)),(10,i)] for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 2]
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 11: # xxxyyy triple sequence
            # return all larger d sequences of same length
            l = int(len(opact)/3)
            out = [[''.join(r2c_base[j]*3 for j in range(i,i+l)),(11,i)] for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 3]
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 12: # xxxyyy triple sequence (plane) + wings of size 1
            # return all larger d sequences of same length
            stls = state2list_1D(Nelems)
            l = int(len(opact)/4)
            triseq = [[''.join(r2c_base[j]*3 for j in range(i,i+l)),(12,i)] for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 3]
            out = []
            for ts in triseq:
                tsls = [t for t in ts[0]]
                counterA = Counter(tsls)
                counterB = Counter(stls)
                others = list((counterB - counterA).elements())
                combinations_string = list(set(combinations(others, l)))
                for comb in combinations_string:
                    out.append([ts[0]+''.join(comb),ts[1]])
            out.extend(all_bombs_new(Nelems))
            #return out
        
        elif opinfo[0] == 13: # xxxyyy triple sequence (plane) + wings of size 2!
            # return all larger d sequences of same length
            stls = state2list_1D(Nelems)
            l = int(len(opact)/5)
            triseq = [[''.join(r2c_base[j]*3 for j in range(i,i+l)),(13,i)] for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 3]
            out = []
            for ts in triseq:
                tsls = [t for t in ts[0]]
                counterA = Counter(tsls)
                counterB = Counter(stls)
                others = list((counterB - counterA).elements())
                counterC = Counter(others)
                comb = []
                for k in list(counterC.keys()):
                    if counterC[k] >= 2:
                        comb.append(k*2)
                        if counterC[k] == 4:
                            comb.append(k*2)
                combinations_string = list(set(combinations(comb, l)))
                for comb in combinations_string:
                    out.append([ts[0]+''.join(comb),ts[1]])
            out.extend(all_bombs_new(Nelems))
            #return out
        if not forcemove:
            out = [['',(0,0)]] + out
        return out
    else: # opponent pass, or it is the first move
        # ALL ACTIONS!
        # Assign type&rank to these actions
        return all_actions(mystate,forcemove)
        #return all_actions_cpp(mystate,forcemove)

def cards2action(cards): # from str representation to action type and rank

    count = Counter(cards)
    values = list(count.values())
    keys = list(count.keys())

    # check for BR and empty
    if cards == '':
        return [cards, (0,0)]
    elif cards == 'BR' or cards == 'RB':
        return [cards, (4,13)]

    # Check for special types of hands
    elif len(values) == 1:  # All cards are the same
        if values[0] == 1:
            return [cards, (1, c2r_base[keys[0]])]  # Single
        elif values[0] == 2:
            return [cards, (2, c2r_base[keys[0]])]  # Double
        elif values[0] == 3:
            return [cards, (3, c2r_base[keys[0]])]  # Triple
        elif values[0] == 4:
            return [cards, (4, c2r_base[keys[0]])]  # Bomb

    elif len(values) == 2:  # Possible 3+1, 3+2, 4+1+1, 4+2, 4+2=2 etc.
        if 3 in values and 1 in values:
            return [cards, (5, c2r_base[keys[values.index(3)]])]  # 3+1
        elif 3 in values and 2 in values:
            return [cards, (6, c2r_base[keys[values.index(3)]])] # 3+2
        elif 4 in values and len(cards) == 6:
            return [cards, (7, c2r_base[keys[values.index(4)]])]  # 4+2 assuming extra are a pair
        elif 4 in values and len(cards) == 8:
            return [cards, (8, max(c2r_base[k] for k in keys))]  # 4+4 as 4+2+2 assuming extra are a pair

    elif len(values) == 3:
        if 4 in values and len(cards) == 6:
            return [cards, (7, c2r_base[keys[values.index(4)]])]  # 4+1+1 assuming extra are 2 singles
        elif 4 in values and len(cards) == 8:
            return [cards, (8, c2r_base[keys[values.index(4)]])]  # 4+2+2

    # Check for sequences and multi-part hands
    if len(keys) >= 5 and (len(keys)==len(cards)) and sorted([c2r_base[k] for k in keys]) == list(range(min([c2r_base[k] for k in keys]), max([c2r_base[k] for k in keys])+1)):
        return [cards, (9, min([c2r_base[k] for k in keys]))]  # Straight sequence

    elif len(cards) >= 6 and len(keys) == len(cards)/2 and max(values) == 2 and sorted([c2r_base[k] for k in keys]) == list(range(min([c2r_base[k] for k in keys]), max([c2r_base[k] for k in keys])+1)):
        return [cards, (10, min([c2r_base[k] for k in keys]))]

    elif len(cards) >= 6 and len(keys) == len(cards)/3 and max(values) == 3 and sorted([c2r_base[k] for k in keys]) == list(range(min([c2r_base[k] for k in keys]), max([c2r_base[k] for k in keys])+1)):
        return [cards, (11, min([c2r_base[k] for k in keys]))]

    # check for triple with wings
    cv = Counter(values)
    #print(cv)
    if len(cards) >= 8 and (cv[3]+cv[4]) == len(cards) / 4: # single wing
        s = sorted([c2r_base[k] for k in keys if count[k] >= 3])
        if len(s) == max(s)-min(s)+1:
            return [cards, (12, min(s))]
    
    elif len(cards) >= 10 and cv[3] == len(cards) / 5 and cv[1] == 0: # pair wing
        s = sorted([c2r_base[k] for k in keys if count[k] == 3])
        #print(s)
        if len(s) == max(s)-min(s)+1:
            return [cards, (13, min(s))]

    return None  # Unknown or more complex pattern

def state2str_1D(Nelems): # no change (the state is the same as Nelems)
    st = ''
    for i in range(15):
        k = r2c_base[i]
        for r in range(int(Nelems[i])):
            st += k
    return st

def emptystate_npy():
    return np.zeros((4,15),dtype=np.float32)

def str2state(st):
    ct = Counter(st)
    state = emptystate_npy()
    for char, count in ct.items():
        state[4-count:, c2r_base[char]] = 1
    return torch.from_numpy(state)

lastmove = cards2action(state2str_1D(laststate))

logging.info(f'Last Action: {lastmove}\n')

acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)

logging.info(f'Actions: {len(acts)}\n')

unavail = played_cards.sum(dim=0)
# get action
Bigstate = torch.cat([player.unsqueeze(0),
                        unavail.unsqueeze(0),
                        played_cards,
                        visible.unsqueeze(0), # new feature
                        torch.full((1, 15), currBotID),
                        torch_hist])
Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
#print(Bigstate)
# generate inputs
hinput = Bigstate.unsqueeze(0)

model_inter, lstm_out = SLM(hinput.to(torch.float32))


model_inter = torch.cat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                model_inter, # upper and lower states
                                #role,
                                lstm_out, # lstm encoded history
                                ],dim=-1)
model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

# get q values and action
output = QV(model_input2).flatten()
Q = torch.max(output)
best_act = acts[torch.argmax(output)]

logging.info(f'Suggest: {best_act}')

cards = best_act[0]
cardlist = []
for c in cards:
    if c not in 'BR':
        r = c2r_base[c]*4
        while r in cardlist or r not in use_info["own"]:
            r += 1
        cardlist.append(r)
    elif c == 'B':
        cardlist.append(52)
    elif c == 'R':
        cardlist.append(53)

logging.info(f'suggest cards: {cardlist} from {use_info["own"]}\n')

print(json.dumps({
        "response": cardlist
    }))


exit()

def randomOut(poker):
    sepRes, res, lenPoker = separate(poker), [], len(poker)
    lenRes, idx = len(sepRes), 0
    score = []
    for tmp in sepRes:
        lenTmp = len(tmp)
        minNum = min(tmp)
        tmpScore = 0
        if lenPoker < 8: # 手上的牌数很少了
            tmpScore = (17-minNum)/lenTmp
        else:
            tmpScore = (17-minNum)*lenTmp
        score += [tmpScore]
    maxScore = max(score)
    maxScoreIdx = [i for i in range(lenRes) if score[i] == maxScore]
    idx = random.choice(maxScoreIdx)

    # idx = random.randint(0, lenRes-1)
    tmp = sepRes[idx] # 只包含点数
    newPoker, singleTmp = ordinalTransfer(poker), list(set(tmp))
    singleTmp.sort()
    for curr in singleTmp:
        tmpCount = tmp.count(curr)
        idx = newPoker.index(curr)
        res += [poker[idx + j] for j in range(tmpCount)]
    #tmpCount = newPoker.count(newPoker[0])
    #res = [[poker[i] for i in range(tmpCount)]]
    return res

if countPass == 2: # 长度为0，自己是地主，随便出；在此之前已有两个pass，上一个轮是自己占大头，不能pass，否则出错失败
    # 有单张先出单张
    #logging.info(f"Before {poker}\n")
    #logging.info(f"Before {lastTypeP}, {lastMP}, {lastSP}\n")
    res = randomOut(poker)
    #logging.info(f"After {poker}\n")
    print(json.dumps({
        "response": res
    }))
    #f.write(f'IN IF COND\n')
    exit()

if currBotID == 1 and countPass == 1: # 上一轮是农民乙出且地主选择pass，为了不压过队友选择pass
    print(json.dumps({
        "response": []
    }))
    exit()


res = searchCard(poker, lastTypeP, lastMP, lastSP)
lenRes = len(res)

'''f.write('Available:\n\n\n')
for r in res:
    f.write(f'{r}')

f.close()'''

if lenRes == 0: # 应当输出pass
    print(json.dumps({
        "response": []
    }))
else:
    pokerOut, typeP = [], "空"
    for i in range(lenRes):
        pokerOut = res[i]
        typeP, _, _ = checkPokerType(pokerOut)
        if typeP != "火箭" and typeP != "炸弹":
            break

    if (currBotID == 2 and countPass == 0) or (currBotID == 1 and countPass == 1): # 两个农民不能起内讧，起码不能互相炸
        if typeP == "火箭" or typeP == "炸弹":
            pokerOut = []
    else: # 其他情况是一定要怼的
        pokerOut = res[0]

    print(json.dumps({
        "response": pokerOut
    }))