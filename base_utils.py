import torch
import random
from collections import Counter
from itertools import combinations
import time
import os
import sys
#from model_V2 import *
import numpy as np

from collections import defaultdict
import search_action_space

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

def emptystate_npy_1D(): # *15 (omitted first axis) instead of 4*15
    return np.zeros(15,dtype=np.float32)

#@profile
def state2str(Nelems):
    st = ''
    for i in range(15):
        k = r2c_base[i]
        for r in range(int(Nelems[i])):
            st += k
    return st

def state2str_1D(Nelems): # no change (the state is the same as Nelems)
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

def state2list_1D(Nelems): # no change (the state is the same as Nelems)
    ls = []
    for i in range(15):
        k = r2c_base[i]
        for r in range(int(Nelems[i])):
            ls.append(k)
    return ls

def state2list_1D(Nelems):
    # Ensure Nelems is a numpy array and r2c_base is appropriately sized
    #Nelems = np.asarray(Nelems)
    #assert len(Nelems) == 15
    #assert len(r2c_base) == 15
    
    # Use numpy.repeat to replicate r2c_base elements according to Nelems
    result = np.repeat(r2c_base_arr, Nelems.astype(int))
    
    return result.tolist()

#@profile
def str2state(st):
    ct = Counter(st)
    state = emptystate_npy()
    for char, count in ct.items():
        state[4-count:, c2r_base[char]] = 1
    return torch.from_numpy(state)

def str2state_compressed(st):
    ct = Counter(st)
    state = np.zeros((1,15),dtype=np.float32)
    for char, count in ct.items():
        state[0, c2r_base[char]] = count
    return torch.from_numpy(state)

def str2state_compressed_1D(st):
    ct = Counter(st)
    state = np.zeros(15,dtype=np.float32)
    for char, count in ct.items():
        state[c2r_base[char]] = count
    return torch.from_numpy(state)

def str2state_1D(st): # same as str2state_compressed_1D
    ct = Counter(st)
    state = np.zeros(15,dtype=np.float32)
    for char, count in ct.items():
        state[c2r_base[char]] = count
    return torch.from_numpy(state)

def str2state_1D_npy(st): # does not convert to torch tensor
    ct = Counter(st)
    state = np.zeros(15,dtype=np.float32)
    for char, count in ct.items():
        state[c2r_base[char]] = count
    return state

# create action based on last state
# create type along with action!
def all_bombs_new(Nelems):
    quad = r2c_base_arr[Nelems==4]
    out = [[Q*4,(4,c2r_base[Q[0]])] for Q in quad]
    if Nelems[-2:].sum() == 2:
        out.append(['BR',(4,13)])
    return out

# create action based on last state
# create type along with action!

#@profile
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

#@profile
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

def all_planes(Nelems, triseq):
    # combinations function is quite fast compared to cpp and easier to use, so plane with wings are searched with python.

    triseq = [(''.join([T*3 for T in r2c_base_arr[i_l[0]:i_l[0]+i_l[1]]]),(11,i_l[0])) for i_l in triseq]
    out = []
    
    for ts in triseq:
        l = int(len(ts[0])/3)

        if l < 6: # 12 plane + wings of size 1
            sub = Nelems.copy()
            sub[ts[1][1]:ts[1][1]+l] -= 3
            sub = state2list_1D(sub)
            combinations_string = list(set(combinations(sub, l)))
            for comb in combinations_string:
                out.append((ts[0]+''.join(comb),(12,ts[1][1])))
        
        if l < 5: # 13 plane + wings of size 2
            sub = Nelems.copy()
            sub[ts[1][1]:ts[1][1]+l] -= 3
            sub = state2list_1D(sub)
            counterC = Counter(sub)
            comb = []
            for k in list(counterC.keys()):
                if counterC[k] >= 2:
                    comb.append(k*2)
                    if counterC[k] == 4:
                        comb.append(k*2)
            combinations_string = list(set(combinations(comb, l)))
            for comb in combinations_string:
                out.append((ts[0]+''.join(comb),(13,ts[1][1])))

    return out

def avail_planes(opinfo, opact, Nelems):
    # combinations function is quite fast compared to cpp and easier to use, so plane with wings are searched with python.
    
    out = []
    if opinfo[0] == 12: # xxxyyy triple sequence (plane) + wings of size 1
        # return all larger d sequences of same length
        stls = state2list_1D(Nelems)
        l = int(len(opact)/4)
        triseq = [(''.join(r2c_base[j]*3 for j in range(i,i+l)),(12,i)) for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 3]
        out = []
        for ts in triseq:
            tsls = [t for t in ts[0]]
            counterA = Counter(tsls)
            counterB = Counter(stls)
            others = list((counterB - counterA).elements())
            combinations_string = list(set(combinations(others, l)))
            for comb in combinations_string:
                out.append((ts[0]+''.join(comb),ts[1]))
        out.extend(all_bombs_new(Nelems))
        #return out
    
    elif opinfo[0] == 13: # xxxyyy triple sequence (plane) + wings of size 2!
        # return all larger d sequences of same length
        stls = state2list_1D(Nelems)
        l = int(len(opact)/5)
        triseq = [(''.join(r2c_base[j]*3 for j in range(i,i+l)),(13,i)) for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 3]
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
                out.append((ts[0]+''.join(comb),ts[1]))
        out.extend(all_bombs_new(Nelems))
    return out

#@profile
def all_actions_cpp(mystate_array,forcemove=0):
    #input_array = mystate.numpy()
    cacts, triseq = search_action_space.get_all_actions(mystate_array)
    if not forcemove:
        cacts.insert(0,('',(0,0)))
    cacts.extend(all_planes(mystate_array,triseq))
    return cacts

#@profile
def avail_actions_cpp(opact,opinfo,mystate,forcemove=0):
    input_array = mystate.numpy()
    if opinfo[0] != 0:
        cacts = search_action_space.get_avail_actions(opact, opinfo, input_array)
        if not forcemove:
            cacts.insert(0,('',(0,0)))
        cacts.extend(avail_planes(opinfo, opact, input_array))
    else:
        cacts = all_actions_cpp(input_array,forcemove)
    return cacts

#@profile
def all_planes_state(Nelems, triseq): # now mystate is the same shape as Nelems
    triseq = [(''.join([T*3 for T in r2c_base_arr[i_l[0]:i_l[0]+i_l[1]]]),(11,i_l[0])) for i_l in triseq]
    out = []
    
    for ts in triseq:
        l = int(len(ts[0])/3)
        ts_state = str2state_1D_npy(ts[0])

        if l < 6: # 12 plane + wings of size 1
            sub = Nelems.copy()
            sub[ts[1][1]:ts[1][1]+l] -= 3
            sub = state2list_1D(sub)

            combinations_string = list(set(combinations(sub, l)))
            combinations_state = np.zeros((len(combinations_string),15),dtype=np.float32) + ts_state[None,:]
            for i in range(l):
                idxi = [c2r_base[_[i]] for _ in combinations_string]
                combinations_state[np.arange(len(combinations_state)),idxi] += 1

            out.extend([(cs,(12,ts[1][1])) for cs in combinations_state])
        
        if l < 5: # 13 plane + wings of size 2
            sub = Nelems.copy()
            sub[ts[1][1]:ts[1][1]+l] -= 3
            sub = state2list_1D(sub)
            counterC = Counter(sub)
            comb = []
            for k in list(counterC.keys()):
                if counterC[k] >= 2:
                    comb.append(k)
                    if counterC[k] == 4:
                        comb.append(k)
            combinations_string = list(set(combinations(comb, l)))
            combinations_state = np.zeros((len(combinations_string),15),dtype=np.float32) + ts_state[None,:]
            for i in range(l):
                idxi = [c2r_base[_[i]] for _ in combinations_string]
                combinations_state[np.arange(len(combinations_state)),idxi] += 2

            out.extend([(cs,(13,ts[1][1])) for cs in combinations_state])


    return out#, out2

#@profile
def avail_planes_state(opinfo, lopact, Nelems):
    out = []
    if opinfo[0] == 12: # xxxyyy triple sequence (plane) + wings of size 1
        # return all larger d sequences of same length
        stls = state2list_1D(Nelems)
        l = int(lopact/4)
        triseq = [(''.join(r2c_base[j]*3 for j in range(i,i+l)),(12,i)) for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 3]
        out = []
        for ts in triseq:
            ts_state = str2state_1D_npy(ts[0])

            tsls = [t for t in ts[0]]
            counterA = Counter(tsls)
            counterB = Counter(stls)
            others = list((counterB - counterA).elements())
            combinations_string = list(set(combinations(others, l)))

            combinations_state = np.zeros((len(combinations_string),15),dtype=np.float32) + ts_state[None,:]
            for i in range(l):
                idxi = [c2r_base[_[i]] for _ in combinations_string]
                combinations_state[np.arange(len(combinations_state)),idxi] += 1

            out.extend([(cs,ts[1]) for cs in combinations_state])

            #for comb in combinations_string:
            #    out_ = (str2state_1D_npy(ts[0]+''.join(comb)),ts[1])
        b = all_bombs_new(Nelems)
        out.extend([(str2state_1D_npy(_[0]),_[1]) for _ in b])
        #return out
    
    elif opinfo[0] == 13: # xxxyyy triple sequence (plane) + wings of size 2!
        # return all larger d sequences of same length
        stls = state2list_1D(Nelems)
        l = int(lopact/5)
        triseq = [(''.join(r2c_base[j]*3 for j in range(i,i+l)),(13,i)) for i in range(opinfo[1]+1,13-l) if Nelems[i:i+l].min() >= 3]
        out = []
        for ts in triseq:
            ts_state = str2state_1D_npy(ts[0])

            tsls = [t for t in ts[0]]
            counterA = Counter(tsls)
            counterB = Counter(stls)
            others = list((counterB - counterA).elements())
            counterC = Counter(others)
            comb = []
            for k in list(counterC.keys()):
                if counterC[k] >= 2:
                    comb.append(k)
                    if counterC[k] == 4:
                        comb.append(k)
            combinations_string = list(set(combinations(comb, l)))

            combinations_state = np.zeros((len(combinations_string),15),dtype=np.float32) + ts_state[None,:]
            for i in range(l):
                idxi = [c2r_base[_[i]] for _ in combinations_string]
                combinations_state[np.arange(len(combinations_state)),idxi] += 2

            out.extend([(cs,ts[1]) for cs in combinations_state])

            #for comb in combinations_string:
            #    out_ = (str2state_1D_npy(ts[0]+''.join(comb)),ts[1])
        b = all_bombs_new(Nelems)
        out.extend([(str2state_1D_npy(_[0]),_[1]) for _ in b])
    return out

def all_actions_cpp_state(mystate,forcemove=0):
    input_array = mystate.numpy()
    cacts, triseq = search_action_space.get_all_actions_array(input_array)
    if not forcemove:
        cacts.insert(0,(np.zeros(15,dtype=np.float32),(0,0)))
    cacts.extend(all_planes_state(input_array,triseq))
    return cacts

def avail_actions_cpp_state(opact,opinfo,mystate,forcemove=0):
    input_array = mystate.numpy()
    if opinfo[0] != 0:
        cacts = search_action_space.get_avail_actions_array(int(opact.sum()), opinfo, input_array)
        if not forcemove:
            cacts.insert(0,(np.zeros(15,dtype=np.float32),(0,0)))
        cacts.extend(avail_planes_state(opinfo, int(opact.sum()), input_array))
    else:
        cacts = all_actions_cpp_state(mystate,forcemove)
    return cacts

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

def init_game_3card(): # landlord has 3 more cards, which are visible to all
    st = '3333444455556666777788889999XXXXJJJJQQQQKKKKAAAA2222BR'
    idx = list(range(54))
    random.shuffle(idx)
    L = ''.join([st[i] for i in idx[:20]])
    B = L[-3:] # the three cards that are visible to all
    U = ''.join([st[i] for i in idx[20:37]])
    D = ''.join([st[i] for i in idx[37:]])
    Lst = str2state_1D(L)
    Ust = str2state_1D(U)
    Dst = str2state_1D(D)
    Bst = str2state_1D(B)
    return [Lst,Ust,Dst,Bst]

def init_game_3card_bombmode(strength=200):
    # Create the deck
    ranks = ['3', '4', '5', '6', '7', '8', '9', 'X', 'J', 'Q', 'K', 'A', '2']
    deck = ['B', 'R'] + [rank for rank in ranks for _ in range(4)]
    
    # Shuffle the deck before distribution
    random.shuffle(deck)
    
    # Create groups
    L, U, D, B = [], [], [], []
    groups = [L, U, D, B]
    
    # Helper to decide group based on existing composition
    group_probabilities = defaultdict(lambda: [1, 1, 1, 0.1])  # Initial probabilities favor M, N, O equally, P less
    
    for card in deck:
        # Adjust probabilities based on card rank and strength parameter
        base_probs = group_probabilities[card]
        adjusted_probs = [base**strength for base in base_probs]
        chosen_group = random.choices(groups, weights=adjusted_probs)[0]
        
        chosen_group.append(card)
        
        # Update probabilities to favor the chosen group for similar cards
        for i in range(4):
            if groups[i] is chosen_group:
                group_probabilities[card][i] += 1

    # Ensure the groups have the required number of cards (M, N, O should have 17, P should have 3)
    all_cards = L + U + D + B
    L, U, D, B = all_cards[:17], all_cards[17:34], all_cards[34:51], all_cards[51:]

    lists = [L, U, D]
    random.shuffle(lists)
    L, U, D = lists

    # Sort each group by the rank defined
    L.sort(key=lambda card: c2r_base[card])
    U.sort(key=lambda card: c2r_base[card])
    D.sort(key=lambda card: c2r_base[card])
    B.sort(key=lambda card: c2r_base[card])
    
    Lst = str2state_1D(''.join(L))
    Ust = str2state_1D(''.join(U))
    Dst = str2state_1D(''.join(D))
    Bst = str2state_1D(''.join(B))
    return [Lst+Bst,Ust,Dst,Bst]

# Example of using the function
Label = ['Landlord','Farmer-0','Farmer-1']

if __name__ == '__main__':

    #L, U, D, B = init_game_3card_bombmode(strength=500)
    #print(f"Group M: {L}")
    #print(f"Group N: {U}")
    #print(f"Group O: {D}")
    #print(f"Group P: {B}")
    
    #a = cards2action('')
    #print(avail_actions(a[0],a[1],str2state('3QQQQK'),1))
    T0, T1 = 0,0
    for i in range(100):
        stri = state2str(init_game_3card()[0].sum(dim=-2))
        t0 = time.time()
        for i in range(100):
            st0 = str2state(stri).sum(dim=-2,keepdims=True)
        t1 = time.time()
        for i in range(100):
            st1 = str2state_compressed(stri)
        t2 = time.time()

        T0 += t1-t0
        T1 += t2-t1
    print(T0,T1)
    print(st0,st1)