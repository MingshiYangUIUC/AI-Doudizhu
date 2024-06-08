# no elo, only win rate

import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader


from model_utils import *
#from base_utils import *
from base_funcs_selfplay import gating_batchpool

from collections import deque
from torch.multiprocessing import Pool
from tqdm import tqdm

import os
import sys

import torch.multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
import gc

wd = os.path.join(os.path.dirname(__file__))


def simulate_match_wrapper(m1, m2, t, device, nh, ng, ne, seed): # dual match
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    #m1, m2, t, device, nh, ng, ne, seed = args
    models = [m1, m2]
    torch.cuda.empty_cache()
    gatingresult = gating_batchpool(models, t, device, Nhistory=nh, ngame=ng, ntask=ne, rseed=seed)
    LW = (gatingresult==0).sum()
    torch.cuda.empty_cache()
    #gatingresult = gating_batchpool(models[::-1], t, device, Nhistory=nh, ngame=ng, ntask=ne, rseed=seed)
    #FW = (np.array(gatingresult)!=0).sum()
    gc.collect()
    return LW #, FW

if __name__ == '__main__':

    eval_device = 'cpu'
    device = 'cpu'

    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    mp.set_start_method('spawn', force=True)
    
    # determine which models participate in the contest
    try:
        nsim_perplayer = int(sys.argv[1])
        pstart = int(sys.argv[2])
        pstop = int(sys.argv[3])
        pstep = int(sys.argv[4])
        players = [str(i).zfill(10) for i in range(pstart,pstop+1,pstep)]
        terminate = False

    except:
        nsim_perplayer = 5000
        players = [str(i).zfill(10) for i in range(25000000,175000000+1,25000000)]
        terminate = True
    
    if terminate:
        print('Arguments incomplete. Please provide n-sim, player-start, player-stop (inclusive), player-step.')
        sys.exit(1)

    nsim = nsim_perplayer
    players.append(players[-1])
    num_processes = min(12,(len(players)-1)*2)

    gate_player_index = len(players)-1 # other players play with this player

    versions = ['H15-V2_2.2']


    models = []
    nplayer = len(players)*len(versions)
    fullplayers = []

    for v in versions:
        fullplayers += players

    for session in players:
        for version in versions:

            # V2_1_2-Contester
            #SLM = Network_Pcard_V1_1(20,5)
            #QV = Network_Qv_Universal_V1_1(6,15,512)
            # V2_2_1
            SLM = Network_Pcard_V2_1(15+7, 7, y=1, x=15, lstmsize=512, hiddensize=1024)
            QV = Network_Qv_Universal_V1_1(6,15,1024)

            SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{version}_{session}.pt')))
            QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{version}_{session}.pt')))

            SLM.eval()
            QV.eval()

            models.append(
                [SLM,QV]
                )
    
    ax_player = list(range(nplayer))

    outfile2 = os.path.join(wd,'data',f'winrates_{"".join(versions)}_{fullplayers[0]}-{fullplayers[-1]}.txt')

    try:
        f = open(outfile2,'r').readlines()
        Fullstat = np.zeros((nplayer,3),dtype=np.int64)
        Fullstat[:,0] = np.array([int(s.split('-')[-3]) for s in f[1:]],dtype=np.int64)
        Fullstat[:,1] = np.array([int(s.split('-')[-2]) for s in f[1:]],dtype=np.int64)
        Fullstat[:,-1] = np.array([int(s.split('-')[-1]) for s in f[1:]],dtype=np.int64)
        #ngames = np.array([int(s.split('-')[-1]) for s in f[1:]],dtype=np.int64)
        csim = int(f[0].split()[1])
    except:
        csim = 0
        Fullstat = np.zeros((nplayer,3),dtype=np.int64)
        pass

    print(csim)
    #print(Fullstat)
    #quit()
    # random seed
    seed = random.randint(-1000000000,1000000000)
    print(seed)

    # create gating test
    args = []
    for i in range(len(players)):
        if i != gate_player_index:
            args.append((models[i], models[gate_player_index], 0,'cuda',15,64,nsim_perplayer,seed))
            args.append((models[gate_player_index], models[i], 0,'cuda',15,64,nsim_perplayer,seed))

    print(len(args),len(args[0]))
    # Create a pool of workers and distribute the tasks
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(simulate_match_wrapper, args, chunksize=len(args)//num_processes)
        #results = list(tqdm(pool.imap_unordered(simulate_match_wrapper, tasks), total=len(tasks)))

    #print(len(results))
    results = np.array(results).reshape(-1,2)
    results[:,1] = nsim_perplayer - results[:,1]
    #print(results)
    #quit()
    laststat = np.sum(nsim_perplayer-results,axis=0)[::-1]
    fullstat = np.append(results, laststat[None,:],axis=0)

    # wingame stats
    f = open(outfile2,'w')
    f.write(f'Nsim {int(csim+nsim)}\n')

    ngames = np.zeros(len(results)+1).astype(np.int64)+nsim
    ngames[-1] = (nsim)*(len(results))
    fullstat = np.append(fullstat,ngames[:,None],axis=1)
    Fullstat += fullstat

    for i in range(nplayer):
        f.write(f'M{str(fullplayers[i]).zfill(3)} - {Fullstat[i][0]} - {Fullstat[i][1]} - {Fullstat[i][2]}\n')
    f.close()

    WRL = Fullstat[:,0] / Fullstat[:,-1]
    WRF = Fullstat[:,1] / Fullstat[:,-1]


    plt.figure(figsize=(10,4))
    plt.title(f'Nsim {int(csim+nsim)}')
    for i,v in enumerate(versions):
        plt.plot(fullplayers[i*len(players):(i+1)*len(players)],WRL)
        plt.plot(fullplayers[i*len(players):(i+1)*len(players)],WRF)
        plt.plot(fullplayers[i*len(players):(i+1)*len(players)],(WRL+WRF)/2)
    plt.axhline(0.5,zorder=-10,alpha=0.6,color='black')
    plt.savefig(os.path.join(wd,'data',f'winrates_{"".join(versions)}_{fullplayers[0]}-{fullplayers[-1]}.png'))
    #plt.show()