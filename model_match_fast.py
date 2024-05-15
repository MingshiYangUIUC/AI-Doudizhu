# no elo, only win rate

# L model vs a pair of D and U


import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader

from base_funcs_V2_1_1 import *
from model_V2 import *

from collections import deque
from torch.multiprocessing import Pool
from tqdm import tqdm

import os
import sys

import torch.multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt

wd = os.path.join(os.path.dirname(__file__))

class EloSystem:
    def __init__(self, num_players=101, initial_rating=1500, k_factor=10):
        self.ratings = [initial_rating] * num_players
        self.k_factor = k_factor

    def expected_score(self, rating_a, rating_b):
        """ Calculate expected score based on current ratings """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, player_a, player_b, score_a):
        """ Update ratings of two players after a match """
        # Calculate expected scores
        expected_a = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        expected_b = 1 - expected_a  # Since it's a zero-sum game

        # Update ratings
        self.ratings[player_a] += self.k_factor * (score_a - expected_a)
        self.ratings[player_b] += self.k_factor * ((1 - score_a) - expected_b)

    def simulate_match(self, player_a, player_b, outcome):
        """ Simulate a match between two players """
        # Outcome is the total score of player_a over two games (0, 0.5, or 1)
        self.update_ratings(player_a, player_b, outcome)


def psuedomatch(player1,player2):
    # result based on difference
    adv = (player1-player2)/(nplayer-1)
    rng = random.uniform(-1,1)

    if adv - rng > 1: # win
        return 1
    elif adv - rng < -1: # lose
        return 0
    else:
        return 0.5


def simEpisode_notrain(Initstates, Models, temperature, verbose=0):
    if Initstates is None:
        Initstates = init_game_3card() # [Lstate, Dstate, Ustate]

    maxhist = 15

    unavail = ''
    history = torch.zeros((maxhist,4,15))
    lastmove = ['',(0,0)]

    Turn = 0
    Npass = 0 # number of pass applying to rules
    Cpass = 0 # continuous pass
    Condition = 0

    Forcemove = True # whether pass is not allowed

    while True: # game loop
        SLM,QV = Models[Turn%3]
        player = Initstates[Turn%3]

        # get action directly
        action, Q = get_action_serial_V2_1_1(Turn, SLM,QV,Initstates,unavail,lastmove, Forcemove, history, temperature)
        
        if Forcemove:
            Forcemove = False

        # conduct a move
        myst = state2str(player.sum(dim=0).numpy())
        cA = Counter(myst)
        cB = Counter(action[0])
        newst = ''.join(list((cA - cB).elements()))
        newunavail = unavail + action[0]
        newhist = torch.roll(history,1,dims=0)
        newhist[0] = str2state(action[0]) # first row is newest, others are moved downward
        
        #newlast = ['',(0,0)]
        play = action[0]
        if action[1][0] == 0:
            play = 'PASS'
            Cpass += 1
            if Npass < 1:
                Npass += 1
            else:
                #print('Clear Action')
                newlast = ['',(0,0)]
                Npass = 0
                Forcemove = True
        else:
            newlast = action
            Npass = 0
            Cpass = 0

        #myst, action[0], newst, newunavail, newhist[0], newlast
        if verbose:
            print(Label[Turn%3], str(Turn).zfill(3), myst.zfill(20).replace('0',' '), play.zfill(20).replace('0',' '), Label[Turn%3], round(Q.item()*100,1), Npass, Cpass)

        # record
        nextstate = str2state(newst)


        # update
        Initstates[Turn%3] = nextstate
        unavail = newunavail
        history = newhist
        lastmove = newlast
        
        if len(newst) == 0:
            Condition = 1
            break

        Turn += 1

    #if Condition == 1:
    if verbose:
        print(f'Player {Label[Turn%3]} Win')
    #print(len(BufferStatesActs[0]),len(BufferRewards[0]))
    #quit()
    return Turn

def simulate_match_wrapper(m1, m2, t, device, nh, ng, ne, seed): # dual match
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    #m1, m2, t, device, nh, ng, ne, seed = args
    models = [m1, m2]
    gatingresult = gating_batchpool(models, t, device, Nhistory=nh, ngame=ng, ntask=ne, rseed=seed)
    LW = (np.array(gatingresult)==0).sum()
    gatingresult = gating_batchpool(models[::-1], t, device, Nhistory=nh, ngame=ng, ntask=ne, rseed=seed)
    FW = (np.array(gatingresult)!=0).sum()

    return LW, FW

if __name__ == '__main__':

    eval_device = 'cpu'
    device = 'cpu'

    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    mp.set_start_method('spawn', force=True)
    
    # determine which models participate in the contest
    nsim_perplayer = 5000
    nsim = nsim_perplayer
    

    players = [str(i).zfill(10) for i in range(0,35000000+1,5000000)]
    players.append(players[-1])
    num_processes = min(12,len(players)-1)

    gate_player_index = len(players)-1 # other players play with this player

    versions = ['H15-V2_1.2-Contester']


    models = []
    nplayer = len(players)*len(versions)
    fullplayers = []

    for v in versions:
        fullplayers += players

    for session in players:
        for version in versions:

            SLM = Network_Pcard_V1_1(20,5)#,Network_Pcard(N_history+4),Network_Pcard(N_history+4)
            QV = Network_Qv_Universal_V1_1(6,15,512)

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

    print(len(args),len(args[0]))
    # Create a pool of workers and distribute the tasks
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(simulate_match_wrapper, args)
        #results = list(tqdm(pool.imap_unordered(simulate_match_wrapper, tasks), total=len(tasks)))


    results = np.array(results)
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
    plt.axhline(0.5,zorder=-10,alpha=0.6,color='black')
    plt.savefig(os.path.join(wd,'data',f'winrates_{"".join(versions)}_{fullplayers[0]}-{fullplayers[-1]}.png'))
    #plt.show()