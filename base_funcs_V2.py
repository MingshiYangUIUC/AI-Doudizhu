import torch
import random
from collections import Counter
from itertools import combinations
import time
import os
import sys
from model_V2 import *
import numpy as np

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
#print(r2c_base_arr)
c2r_base = {r2c_base[k]:k for k in r2c_base.keys()}

#@profile
def emptystate():
    return torch.zeros((4,15))

def emptystate_npy():
    return np.zeros((4,15),dtype=np.float32)

#@profile
def state2str(Nelems):
    st = ''
    for i in range(15):
        k = r2c_base[i]
        for r in range(int(Nelems[i])):
            st += k
    return st

#@profile
def state2list(state):
    ls = []
    for i in range(15):
        for j in range(4):
            if state[j][i]:
                ls.append(r2c_base[i])
    return ls

#@profile
def state2list2(Nelems):
    ls = []
    for i in range(15):
        k = r2c_base[i]
        for r in range(int(Nelems[i])):
            ls.append(k)
    return ls

#@profile
def str2state(st):
    ct = Counter(st)
    state = emptystate_npy()
    for char, count in ct.items():
        state[4-count:, c2r_base[char]] = 1
    return torch.from_numpy(state)

# create action based on last state
# create type along with action!
def all_bombs_new(Nelems):
    quad = r2c_base_arr[Nelems==4]
    out = [[Q*4,(4,c2r_base[Q[0]])] for Q in quad]
    if Nelems[-2:].sum() == 2:
        out += [['BR',(4,13)]]
    return out

# create action based on last state
# create type along with action!

#@profile
def all_actions(mystate,forcemove=0):
    #mystate = str2state('3334555567777XXJ22')
    # opinfo is (type, rank)
    # action is string of stuff '34567' or 'BR' for black.red joker
    # return as "action, type, rank"
    
    # return all

    # 0 pass
    Nelems = mystate.sum(dim=0).numpy()

    if not forcemove:
        out = [['',(0,0)]]
    else:
        out = []

    # 1 single
    sin = r2c_base_arr[Nelems>0]
    LenS = len(sin)
    idx_S = np.arange(15)[Nelems>0]
    out += [[S,(1,c2r_base[S])] for S in sin]

    # 2 double
    dbl = r2c_base_arr[Nelems>=2]
    LenD = len(dbl)
    idx_D = np.arange(15)[Nelems>=2]
    out += [[D*2,(2,c2r_base[D[0]])] for D in dbl]

    # 3 triple
    tri = r2c_base_arr[Nelems>=3]
    out += [[T*3,(3,c2r_base[T[0]])] for T in tri]

    # 4 bomb
    quad = r2c_base_arr[Nelems==4]
    out += [[Q*4,(4,c2r_base[Q[0]])] for Q in quad]
    if Nelems[-2:].sum() == 2:
        out += [['BR',(4,13)]]

    # 5 3+1
    out += [[T*3 + S,(5,c2r_base[T[0]])] for T in tri for S in sin if T != S]

    # 6 3+2
    out += [[T*3 + D*2,(6,c2r_base[T[0]])] for T in tri for D in dbl if T != D] # use existing tri

    # 7 4+1+1
    #quad = [[r2c_base[i]*4,(7,i)] for i in range(15) if Nelems[i] >= 4]
    #quad = r2c_base_arr[Nelems==4]
    #sin1 = [r2c_base[i]*1 for i in range(15) if Nelems[i] >= 1]
    for i in range(LenS):
        for j in range(i+1):
            if i != j or Nelems[idx_S[i]] >= 2:#mystate[:,c2r_base[s1]].sum() >= 2:
                s1,s2 = sin[i], sin[j]
                #print(i,j,s1,s2)
                out += [[Q*4+s1+s2,(7,c2r_base[Q])] for Q in quad if Q != s1 if Q != s2]
    
    # 8 4+2+2
    #dbl1 = [r2c_base[i]*2 for i in range(15) if mystate[:,i].sum() >= 2]
    #for d1 in dbl1:
    #    for d2 in dbl1:
    for i in range(LenD):
        for j in range(i+1):
            if i != j or Nelems[idx_D[i]] >= 4:
                d1,d2 = dbl[i], dbl[j]
                #print(d1,d2,i,j,Nelems,Nelems[idx_D[i]])
                out += [[Q*4+d1*2+d2*2,(8,c2r_base[Q])] for Q in quad if Q != d1 if Q != d2]

    #print(out)
    #quit()
    # 9 sequence
    # for all element find seq
    i = 0
    #while i < 8:
    for i in range(0,8):
        l = 5
        if 0 not in Nelems[i:i+l]:
            out += [[''.join(r2c_base_arr[i:i+l]),(9,i)]]
            #print(i,l,Nelems[l],i+l)
            while Nelems[i+l] != 0 and i+l<12:
                l += 1
                out += [[''.join(r2c_base_arr[i:i+l]),(9,i)]]
        #    i += 1
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
            out += [[''.join([D*2 for D in r2c_base_arr[i:i+l]]),(10,i)]]
            #print(i,l,Nelems[l],i+l)
            while Nelems[i+l] >= 2 and i+l<12:
                l += 1
                out += [[''.join([D*2 for D in r2c_base_arr[i:i+l]]),(10,i)]]
        #    i += 1
        #else:
        #    i += np.argmin(Nelems[i:i+l])+1

    # 11 triple seq
    triseq = []
    #i = 0
    for i in range(0,10):
    #while i < 11:
        l = 2
        if Nelems[i:i+l].min()>=3:
            triseq += [[''.join([T*3 for T in r2c_base_arr[i:i+l]]),(11,i)]]
            #print(i,l,Nelems[l],i+l)
            while Nelems[i+l] >= 3 and i+l<12:
                l += 1
                triseq += [[''.join([T*3 for T in r2c_base_arr[i:i+l]]),(11,i)]]
        #    i += 1
        #else:
        #    i += np.argmin(Nelems[i:i+l])+1

    out += triseq
    #print(out)
    #quit()
    stls = state2list2(Nelems)
    
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
    #print(out)
    #quit()
    return out

#@profile
def avail_actions(opact,opinfo,mystate,forcemove=0):
    # opinfo is (type, rank)
    # action is string of stuff '34567' or 'BR' for black.red joker
    # return as "action, type, rank"
    
    if opinfo[0] != 0:

        Nelems = mystate.sum(dim=0).numpy()

        # Find action based on previous state
        # Assign unique type to new actions
        
        if opinfo[0] == 1: # single
            # return all larger single or bombs
            out = [[r2c_base[i],(1,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 1]
            #sin = r2c_base_arr[Nelems>0]
            #out = [[S,(1,c2r_base[S])] for S in sin if c2r_base[S] > opinfo[1]]
            out += all_bombs_new(Nelems)
            #out += all_bombs(mystate)
            #return out
        
        elif opinfo[0] == 2: # double
            # return all larger double or bombs
            out = [[r2c_base[i]*2,(2,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 2]
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 3: # triple
            # return all larger double or bombs
            out = [[r2c_base[i]*3,(3,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 3]
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 4: # bomb
            # return all larger bombs
            out = [[r2c_base[i]*4,(4,i)] for i in range(opinfo[1]+1,15) if Nelems[i] == 4]

            if Nelems[-2:].sum() == 2:
                #return out + [['BR',(4,13)]]
                out += [['BR',(4,13)]]
            else:
                #return out
                pass
        
        elif opinfo[0] == 5: # 3+1
            # return all larger trio + any 1 and bombs
            tri = [[r2c_base[i]*3,(5,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 3]
            #sin = [r2c_base[i] for i in range(15) if mystate[-1][i] >= 1]
            sin = r2c_base_arr[Nelems>0]
            out = [[t[0] + s,t[1]] for t in tri for s in sin if t[0][0] != s]
            out += all_bombs_new(Nelems)
            #return out

        elif opinfo[0] == 6: # 3+2
            # return all larger trio + any 2 and bombs
            tri = [[r2c_base[i]*3,(6,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 3]
            dbl = r2c_base_arr[Nelems>1]
            #dbl = [r2c_base[i]*2 for i in range(15) if mystate[:,i].sum() >= 2]
            out = [[t[0] + d*2,t[1]] for t in tri for d in dbl if t[0][0] != d]
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 7: # 4 + 1 + 1
            # return all larger trio + any 1+1 and bombs
            quad = [[r2c_base[i]*4,(7,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 4]
            sin1 = r2c_base_arr[Nelems>0]
            out = []
            for s1 in sin1:
                for s2 in sin1:
                    if c2r_base[s1] < c2r_base[s2] or Nelems[c2r_base[s1]] >= 2:
                        out += [[t[0]+s1+s2,t[1]] for t in quad if t[0][0] != s1 if t[0][0] != s2]
            out += all_bombs_new(Nelems)
            #return out

        elif opinfo[0] == 8: # 4 + 2 + 2
            # return all larger trio + any other 1+1 and bombs
            quad = [[r2c_base[i]*4,(8,i)] for i in range(opinfo[1]+1,15) if Nelems[i] >= 4]
            dbl1 = r2c_base_arr[Nelems>1]
            out = []
            for d1 in dbl1:
                for d2 in dbl1:
                    if c2r_base[d1] < c2r_base[d2] or Nelems[c2r_base[d1]] >= 4:
                        out += [[t[0]+d1*2+d2*2,t[1]] for t in quad if t[0][0] != d1 if t[0][0] != d2]
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 9: # abcde sequence
            # return all larger sequences of same length
            l = len(opact)
            out = [[''.join(r2c_base[j] for j in range(i,i+l)),(9,i)] for i in range(opinfo[1]+1,13-l) if mystate[-1,i:i+l].sum() == l]
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 10: # xxyyzz double sequence
            # return all larger d sequences of same length
            l = int(len(opact)/2)
            out = [[''.join(r2c_base[j]*2 for j in range(i,i+l)),(10,i)] for i in range(opinfo[1]+1,13-l) if mystate[-2:,i:i+l].sum() == l*2]
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 11: # xxxyyy triple sequence
            # return all larger d sequences of same length
            l = int(len(opact)/3)
            out = [[''.join(r2c_base[j]*3 for j in range(i,i+l)),(11,i)] for i in range(opinfo[1]+1,13-l) if mystate[-3:,i:i+l].sum() == l*3]
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 12: # xxxyyy triple sequence (plane) + wings of size 1
            # return all larger d sequences of same length
            stls = state2list2(Nelems)
            l = int(len(opact)/4)
            triseq = [[''.join(r2c_base[j]*3 for j in range(i,i+l)),(12,i)] for i in range(opinfo[1]+1,13-l) if mystate[-3:,i:i+l].sum() == l*3]
            out = []
            for ts in triseq:
                tsls = [t for t in ts[0]]
                counterA = Counter(tsls)
                counterB = Counter(stls)
                others = list((counterB - counterA).elements())
                combinations_string = list(set(combinations(others, l)))
                for comb in combinations_string:
                    out.append([ts[0]+''.join(comb),ts[1]])
            out += all_bombs_new(Nelems)
            #return out
        
        elif opinfo[0] == 13: # xxxyyy triple sequence (plane) + wings of size 2!
            # return all larger d sequences of same length
            stls = state2list2(Nelems)
            l = int(len(opact)/5)
            triseq = [[''.join(r2c_base[j]*3 for j in range(i,i+l)),(13,i)] for i in range(opinfo[1]+1,13-l) if mystate[-3:,i:i+l].sum() == l*3]
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
            out += all_bombs_new(Nelems)
            #return out
        if not forcemove:
            out = [['',(0,0)]] + out
        return out
    else: # opponent pass, or it is the first move
        # ALL ACTIONS!
        # Assign type&rank to these actions
        return all_actions(mystate,forcemove)


def cards2action(cards):
    from collections import Counter

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

    elif len(values) == 2:  # Possible 3+1, 3+2, 4+1+1, 4+2, etc.
        if 3 in values and 1 in values:
            return [cards, (5, c2r_base[keys[values.index(3)]])]  # 3+1
        elif 3 in values and 2 in values:
            return [cards, (6, c2r_base[keys[values.index(3)]])] # 3+2
        elif 4 in values and len(cards) == 6:
            return [cards, (7, c2r_base[keys[values.index(4)]])]  # 4+2 assuming extra are a pair

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


def init_game_3card(): # landlord has 3 more cards, which are visible to all
    st = '3333444455556666777788889999XXXXJJJJQQQQKKKKAAAA2222BR'
    idx = list(range(54))
    random.shuffle(idx)
    L = ''.join([st[i] for i in idx[:20]])
    B = L[-3:] # the three cards that are visible to all
    U = ''.join([st[i] for i in idx[20:37]])
    D = ''.join([st[i] for i in idx[37:]])
    Lst = str2state(L)
    Ust = str2state(U)
    Dst = str2state(D)
    Bst = str2state(B)
    return [Lst,Ust,Dst,Bst]

#@profile
def simEpisode_batchpool_softmax(Models, temperature, selfplay_device, Nhistory=6, ngame=20, ntask=100):
    #print('Init wt',Models[0].fc2.weight.data[0].mean())
    #quit()
    Models[0].to(selfplay_device)  # SL
    Models[-1].to(selfplay_device) # QM

    Index = torch.arange(ngame)
    Active = [True for _ in range(ngame)]
    #print(max(Active))

    Init_states = [init_game_3card() for _ in range(ngame)] # [Lstate, Dstate, Ustate]
    Visible_states = [ist[-1] for ist in Init_states] # the 3 card matrix

    unavail = ['' for _ in range(ngame)]
    history = [torch.zeros((Nhistory,4,15)) for _ in range(ngame)]
    lastmove = [['',(0,0)] for _ in range(ngame)]
    newlast = [[] for _ in range(ngame)]

    Turn = torch.zeros(ngame).to(torch.int32)
    Npass = [0 for _ in range(ngame)] # number of pass applying to rules
    Cpass = [0 for _ in range(ngame)] # continuous pass

    Forcemove = [True for _ in range(ngame)] # whether pass is not allowed

    BufferStatesActs = [[[],[],[]] for _ in range(ngame)] # states_actions for 3 players
    BufferRewards = [[[],[],[]] for _ in range(ngame)] # rewards for 3 players

    Full_output = []
    
    processed = ngame

    #Games = ['' for _ in range(ngame)]
    SL_X = [] # no order, ok to use one list
    SL_Y = []

    while processed < ntask or True in Active: # game loop
        # serially get all actions and states

        model_inputs = []
        model_idxs = []
        acts_list = []
        bstates_list = []
        sl_y = []
        
        model = Models[0]

        for iin, _ in enumerate(Index[Active]):

            Tidx = Turn[_]%3
            #playerstate, model = Init_states[_][Tidx], Models[Tidx] # Same model should be used for all Active
            playerstate = Init_states[_][Tidx]
            upper_state = Init_states[_][:-1][(Tidx-1)%3].sum(dim=0)
            lower_state = Init_states[_][:-1][(Tidx+1)%3].sum(dim=0)
            sl_y.append(torch.cat((upper_state,lower_state)))
            #print(Turn)
            visible = Visible_states[_] # 3 cards from landlord, fixed for one full game

            # get card count
            card_count = [int(p.sum()) for p in Init_states[_]]
            CC = torch.zeros((4,15))
            CC[0][:min(card_count[0],15)] = 1
            CC[1][:min(card_count[1],15)] = 1
            CC[2][:min(card_count[2],15)] = 1

            # get actions
            acts = avail_actions(lastmove[_][0],lastmove[_][1],playerstate,Forcemove[_])

            # add states and visible to big state
            Bigstate = torch.cat([playerstate.unsqueeze(0),
                                  str2state(unavail[_]).unsqueeze(0),
                                  CC.unsqueeze(0),
                                  visible.unsqueeze(0), # new feature
                                  history[_]])

            # generate inputs
            hinput = Bigstate.unsqueeze(0)

            model_inputs.append(hinput)
            model_idxs.append(len(acts))
            acts_list.append(acts)

            #bstates_list.append(Bigstate)

        # use all data to run model
        torch.cuda.empty_cache()

        model_inputs = torch.concat(model_inputs)

        # predict state (SL)
        model_inter = model(
            model_inputs.to(selfplay_device)
            ).to('cpu')
        #print(model_inputs.shape)
        SL_X.append(model_inputs.clone().detach())
        SL_Y.append(torch.stack(sl_y))
        #print(model_inter.shape,torch.stack(sl_y).shape)

        role = torch.zeros((model_inter.shape[0],15)) + Tidx

        model_inter = torch.concat([model_inputs[:,0].sum(dim=-2),model_inter,role],dim=-1)
        model_input2 = []

        for i, mi in enumerate(model_inter):
            input_i = torch.stack([torch.cat((mi,str2state(a[0]).sum(dim=0))) for a in acts_list[i]])
            model_input2.append(input_i)
            #bstates_list.append(Bigstate)
        #print(model_inter.shape)
        #quit()
        model_output = Models[-1](torch.cat(model_input2).to(selfplay_device)).to('cpu').flatten()

        # conduct actions for all instances
        for iout, _ in enumerate(Index[Active]):
            Tidx = Turn[_]%3
            playerstate = Init_states[_][Tidx]
            #Bigstate = bstates_list[iout]

            idx_start = sum(model_idxs[:iout])
            idx_end = sum(model_idxs[:iout+1])

            # get q values
            output = model_output[idx_start:idx_end].clone().detach()
            acts = acts_list[iout]

            if temperature == 0:
                q = torch.max(output)
                best_act = acts[torch.argmax(output)]
            else:
                # get action using probabilistic approach and temperature
                probabilities = torch.softmax(output / temperature, dim=0)
                distribution = torch.distributions.Categorical(probabilities)
                q = distribution.sample()
                best_act = acts[q]
            
            #print(torch.argmax(output))
            action = best_act
            
            if Forcemove[_]:
                Forcemove[_] = False

            # conduct a move
            myst = state2str(playerstate.sum(dim=0).numpy())
            cA = Counter(myst)
            cB = Counter(action[0])
            newst = ''.join(list((cA - cB).elements()))
            newunavail = unavail[_] + action[0]
            newhist = torch.roll(history[_],1,dims=0)
            newhist[0] = str2state(action[0]) # first row is newest, others are moved downward

            if action[1][0] == 0:
                Cpass[_] += 1
                if Npass[_] < 1:
                    Npass[_] += 1
                else:
                    newlast[_] = ['',(0,0)]
                    Npass[_] = 0
                    Forcemove[_] = True
            else:
                newlast[_] = action
                Npass[_] = 0
                Cpass[_] = 0

            # record and update
            nextstate = str2state(newst)
            #print(newst)

            BufferStatesActs[_][Tidx].append(torch.concat([model_inter[iout].clone().detach(),newhist[0].sum(dim=0).clone().detach()]).unsqueeze(0))
            BufferRewards[_][Tidx].append(0)

            Init_states[_][Tidx] = nextstate
            unavail[_] = newunavail
            history[_] = newhist
            lastmove[_] = newlast[_]

            if len(newst) == 0:
                BufferRewards[_][Tidx] = [torch.as_tensor(BufferRewards[_][Tidx],dtype=torch.float32)+1]
                if Tidx == 1:
                    BufferRewards[_][Tidx+1] = [torch.as_tensor(BufferRewards[_][Tidx+1],dtype=torch.float32)+1]
                    BufferRewards[_][Tidx-1] = [torch.as_tensor(BufferRewards[_][Tidx-1],dtype=torch.float32)]
                elif Tidx == 2:
                    BufferRewards[_][Tidx-1] = [torch.as_tensor(BufferRewards[_][Tidx-1],dtype=torch.float32)+1]
                    BufferRewards[_][Tidx-2] = [torch.as_tensor(BufferRewards[_][Tidx-2],dtype=torch.float32)]
                elif Tidx == 0:
                    BufferRewards[_][Tidx+1] = [torch.as_tensor(BufferRewards[_][Tidx+1],dtype=torch.float32)]
                    BufferRewards[_][Tidx+2] = [torch.as_tensor(BufferRewards[_][Tidx+2],dtype=torch.float32)]
                
                Active[_] = False

                # send data to output collection
                try:
                    SA = [[torch.concat(p)] for p in BufferStatesActs[_]]
                    Full_output.append([SA, BufferRewards[_], True])
                except:
                    Full_output.append([None,None,False])

        Turn += 1 # all turn += 1

        # if completed games < ntask, reset state for this index, WHEN NEXT TURN IS LANDLORD
        # if processed < ntask:
        for iout, _ in enumerate(Index):
            while not max(Active) and Turn[_]%3 != 0: # if all games stopped and not landlord turn: just add 1 to turn
                Turn += 1
            if processed < ntask and not Active[_] and Turn[_]%3 == 0: # landlord turn, new game
                # clear buffer
                BufferStatesActs[_] = [[],[],[]]
                BufferRewards[_] = [[],[],[]]

                # reset game
                Active[_] = True
                Turn[_] = 0
                Init_states[_] = init_game_3card()
                Visible_states[_] = Init_states[_][-1]
                unavail[_] = ''
                lastmove[_] = ['',(0,0)]
                newlast[_] = []
                Npass[_] = 0
                Cpass[_] = 0
                Forcemove[_] = True
                processed += 1
                print(str(processed).zfill(5),'/', str(ntask).zfill(5), '   ',end='\r')
        #print(Active)
        #print(processed, ntask, Turn)

    return Full_output, torch.cat(SL_X), torch.cat(SL_Y)


if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader, ConcatDataset


    wd = os.path.dirname(__file__)
    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    v_M = 'H15-V5.0_0001000000'
    N_history = int(v_M[1:3])

    SLM = Network_Pcard(N_history+4)#,Network_Pcard(N_history+4),Network_Pcard(N_history+4)

    #LM.load_state_dict(torch.load(os.path.join(wd,'models',f'LM_{v_M}.pt')))
    #DM.load_state_dict(torch.load(os.path.join(wd,'models',f'DM_{v_M}.pt')))
    #UM.load_state_dict(torch.load(os.path.join(wd,'models',f'UM_{v_M}.pt')))

    SLM.eval()
    #DM.eval()
    #UM.eval()

    QV = Network_Qv_Universal(5)

    QV.eval()

    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


    Xbuffer = [[],[],[]]
    Ybuffer = [[],[],[]]
    N_episodes = 100
    chunk, SL_X, SL_Y = simEpisode_batchpool_softmax([SLM,QV], 0, 'cuda', Nhistory=15, ngame=20, ntask=N_episodes)
    '''Lwin = 0
    for BufferStatesActs, BufferRewards, success in chunk:
        if success:
            for i in range(3):
                #print(BufferStatesActs[i].shape)
                Xbuffer[i].extend(BufferStatesActs[i])
                Ybuffer[i].extend(BufferRewards[i])
                if i == 0:
                    #print(len(Xbuffer[0]),len(Ybuffer[0]))
                    Lwin += BufferRewards[i][0][0]
    print([len(_) for _ in Ybuffer],'L WR:',round((Lwin/N_episodes).item()*100,2))
    print(SL_X.shape,SL_Y.shape)

    train_dataset = TensorDataset(torch.concat(list(Xbuffer[0])), torch.concat(list(Ybuffer[0])).unsqueeze(1))
    print(len(train_dataset),torch.concat(list(Xbuffer[0])).shape,torch.concat(list(Ybuffer[0])).unsqueeze(1).shape)
    train_dataset = TensorDataset(torch.concat(list(Xbuffer[1])), torch.concat(list(Ybuffer[1])).unsqueeze(1))
    print(len(train_dataset))
    train_dataset = TensorDataset(torch.concat(list(Xbuffer[2])), torch.concat(list(Ybuffer[2])).unsqueeze(1))
    print(len(train_dataset))'''