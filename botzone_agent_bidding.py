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
import logging

# 红桃 方块 黑桃 草花
# 3 4 5 6 7 8 9 10 J Q K A 2 joker & Joker
# (0-h3 1-d3 2-s3 3-c3) (4-h4 5-d4 6-s4 7-c4) …… 52-joker->16 53-Joker->17

def n2t(n): # convert card to tensor index
    if n < 52:
        return n // 4
    else:
        return n - 52 + 13

def evalAndBid(poker, maxScore):
    
    class BidModel(nn.Module):
        def __init__(self, input_size=15, hidden_size=128, num_hidden_layers=4, output_size=1):
            super(BidModel, self).__init__()
            
            # Define the input layer
            self.input_layer = nn.Linear(input_size, hidden_size)
            
            # Define the hidden layers
            self.hidden_layers = nn.ModuleList()
            for _ in range(num_hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            
            # Define the output layer
            self.output_layer = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # Apply the input layer with ReLU activation
            x = F.relu(self.input_layer(x))
            
            # Apply each hidden layer with ReLU activation
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            
            # Apply the output layer
            x = torch.sigmoid(self.output_layer(x))
            
            return x
        
    EV = BidModel(input_size=15, hidden_size=128, num_hidden_layers=5, output_size=1)

    data_dir = os.path.join(os.getcwd(), 'data')
    EV.load_state_dict(torch.load(os.path.join(data_dir,f'EV_best.pt')))
    EV.eval()
    # construct input state
    player = torch.zeros((15),dtype=torch.int8)
    for n in poker:
        player[n2t(n)] += 1
    
    q = float(EV(player.to(torch.float32).unsqueeze(0)).flatten())
    
    #evals = [i for i in range(maxScore+1, 4)]
    #evals += [0]
    
    # simple strategy
    # score 0: q < 0.5
    # score 1: 0.67 > q > 0.5
    # score 2: 0.83 > q > 0.67
    # score 3: q > 0.83
    if q < 0.5:
        score_q = 0
    elif q < 0.6333:
        score_q = 1
    elif q < 0.8666:
        score_q = 2
    else:
        score_q = 3
    #score_q = min(float(q) // 0.25,3)
    if score_q == 0 or score_q > maxScore:
        return score_q
    else:
        return 0

def printBid(poker, bidHistory):
    bidHistory += [0]
    maxScore = max(bidHistory)
    if len(full_input["requests"]) == 1:
        print(json.dumps({
            "response": evalAndBid(poker, maxScore)
        }))

def getPointFromSerial(x):
    return int(x/4) + 3 + (x==53)

def convertToPointList(poker):
    return [getPointFromSerial(i) for i in poker]

_online = os.environ.get("USER", "") == "root"

if _online:
    full_input = json.loads(input())
else:
    with open("botlogs.json") as fo:
        full_input = json.load(fo)

bid_info = full_input["requests"][0]
if "bid" in bid_info and len(full_input["requests"]) == 1:
    printBid(bid_info["own"], bid_info["bid"])
    exit(0)

if "bid" in bid_info and len(full_input["requests"]) > 1:
    user_info = full_input["requests"][1]
    my_history = full_input["responses"][1:]
    others_history = full_input["requests"][2:]
else:
    user_info = bid_info
    my_history = full_input["responses"]
    others_history = full_input["requests"][1:]

# 手牌，公共牌，地主位置，当前我的bot位置，最终叫牌底分
poker, publiccard, landlordPos, currBotPos, finalBid = user_info["own"], \
    user_info["publiccard"], user_info["landlord"], user_info["pos"], user_info["finalbid"]
if landlordPos == currBotPos:
    poker.extend(publiccard)
    currBotID = 0
else:
    if currBotPos < landlordPos:
        currBotID = int(currBotPos+3-landlordPos)
    else:
        currBotID = int(currBotPos-landlordPos)

history = user_info["history"]
last_history = full_input["requests"][-1]["history"]

start = (landlordPos + 3 - currBotPos) % 3
#print(start)
#print(history)

history = history[(2 - currBotID):]



for i in range(len(my_history)):
    #print('m',my_history[i])
    #print('o',others_history[i]["history"])
    history.append(my_history[i])
    history += others_history[i]["history"]

lenHistory = len(history)


for tmp in my_history:
    for j in tmp:
        poker.remove(j)
poker.sort() # 用0-53编号的牌



#print(history)


#f = open(os.path.join('logs.txt'),'a')


'''# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, 'print.txt')

# Configure the logger
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(message)s')'''


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

#print(torch_hist)

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
            if i != j: # Botzone rule
                s1,s2 = sin[i], sin[j]
                #print(i,j,s1,s2)
                out.extend([[Q*4+s1+s2,(7,c2r_base[Q])] for Q in quad if Q != s1 if Q != s2])
    
    # 8 4+2+2
    #dbl1 = [r2c_base[i]*2 for i in range(15) if mystate[:,i].sum() >= 2]
    #for d1 in dbl1:
    #    for d2 in dbl1:
    for i in range(LenD):
        for j in range(i+1):
            if i != j: # Botzone rule
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
        if l < 6: # 12 plane + wings of size 1 no repeat
            tsls = [t for t in ts[0]]
            counterA = Counter(tsls)
            counterB = Counter(stls)
            others = list((counterB - counterA).elements())
            combinations_string = list(set(combinations(others, l)))
            for comb in combinations_string:
                if len(set(comb)) == l: # no repeat wing
                    passcheck = True
                    for c in comb:
                        if c in ts[0]: # wing not in plane
                            passcheck=False
                    if passcheck:
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
                if len(set([c[0] for c in comb])) == l:
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
                    if c2r_base[s1] < c2r_base[s2]: # Botzone rule
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
                    if c2r_base[d1] < c2r_base[d2]: # Botzone rule
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
                    if len(set(comb)) == l: # no repeat wing
                        passcheck = True
                        for c in comb:
                            if c in ts[0]: # wing not in plane
                                passcheck=False
                        if passcheck:
                            out.append([ts[0]+''.join(comb),(12,ts[1])])
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
                    if len(set([c[0] for c in comb])) == l:
                        out.append([ts[0]+''.join(comb),(13,ts[1][1])])
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

#logging.info(f'Last Action: {lastmove}\n')

acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)

#logging.info(f'Actions: {len(acts)}\n')

# directly pass if acts only include pass
if len(acts) <= 1 and acts[0][0] == '':
    cardlist = []
    print(json.dumps({
        "response": cardlist
    }))

# call model only if eval is needed
else:
    #import time
    #t0 = time.time()
    
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

    try:
        data_dir = os.path.join(os.getcwd(), 'data')
        SLM.load_state_dict(torch.load(os.path.join(data_dir,f'SLM_H15-V2_3.0_best.pt')))
        QV.load_state_dict(torch.load(os.path.join(data_dir,f'QV_H15-V2_3.0_best.pt')))
    except:
        data_dir = os.path.join(os.getcwd(), 'data')
        SLM.load_state_dict(torch.load(os.path.join(data_dir,f'SLM_H15-V2_3.0_0100000000.pt')))
        QV.load_state_dict(torch.load(os.path.join(data_dir,f'QV_H15-V2_3.0_0100000000.pt')))

    SLM.eval()
    QV.eval()

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

    with torch.no_grad():
        model_inter, lstm_out = SLM(hinput.to(torch.float32))


    model_inter = torch.cat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                    model_inter, # upper and lower states
                                    #role,
                                    lstm_out, # lstm encoded history
                                    ],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values and action
    with torch.no_grad():
        output = QV(model_input2).flatten()
    Q = torch.max(output)
    best_act = acts[torch.argmax(output)]

    #logging.info(f'Suggest: {best_act}')
    #print(best_act[0])
    cards = best_act[0]
    cardlist = []
    for c in cards:
        if c not in 'BR':
            r = c2r_base[c]*4
            weights = [0,0,0,0]
            cards = [i for i in range(r,r+4)]
            for x in range(4):
                if r+x not in cardlist and r+x in user_info["own"]:
                    if r+x in user_info["publiccard"]:
                        weights[x] = 10
                    else:
                        weights[x] = 1
            rc = random.choices(cards, weights=weights, k=1)[0]
            #while r in cardlist or r not in use_info["own"]:
            #    r += 1
            cardlist.append(rc)
        elif c == 'B':
            cardlist.append(52)
        elif c == 'R':
            cardlist.append(53)
    #t1 = time.time()
    #logging.info(f'N actions: {len(acts)}, time: {round(t1-t0,2)}s \n')
    #logging.info(f'suggest cards: {cardlist} from {use_info["own"]}\n')
    cardlist.sort()
    print(json.dumps({
            "response": cardlist
        }))

