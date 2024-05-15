# L model vs a pair of D and U


import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader

from base_funcs_V2 import *
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


def get_action_serial_V2_1_1(Turn, SLM,QV,Initstates,unavail,lastmove, Forcemove, history, temperature):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            str2state(unavail).unsqueeze(0),
                            CC.unsqueeze(0),
                            visible.unsqueeze(0), # new feature
                            torch.zeros_like(visible).unsqueeze(0) + Turn%3, # role feature
                            history])

    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    # get all actions
    acts = avail_actions(lastmove[0],lastmove[1],Bigstate[0],Forcemove)
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                hinput[:,5].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    #print(output)
    if temperature == 0:
        Q = torch.max(output)
        best_act = acts[torch.argmax(output)]
    else:
        # get action using probabilistic approach and temperature
        probabilities = torch.softmax(output / temperature, dim=0)
        distribution = torch.distributions.Categorical(probabilities)
        
        q = distribution.sample()
        best_act = acts[q]
        Q = output[q]
    
    action = best_act
    return action, Q

def get_action_serial_V2_0_0(Turn, SLM, QV, Initstates,unavail,lastmove, Forcemove, history, temperature):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            str2state(unavail).unsqueeze(0),
                            CC.unsqueeze(0),
                            visible.unsqueeze(0), # new feature
                            history])

    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    # get all actions
    acts = avail_actions(lastmove[0],lastmove[1],Bigstate[0],Forcemove)
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    #print(output)
    if temperature == 0:
        Q = torch.max(output)
        best_act = acts[torch.argmax(output)]
    else:
        # get action using probabilistic approach and temperature
        probabilities = torch.softmax(output / temperature, dim=0)
        distribution = torch.distributions.Categorical(probabilities)
        
        q = distribution.sample()
        best_act = acts[q]
        Q = output[q]
    
    action = best_act
    return action, Q

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
        if SLM.non_hist == 5:
            action, Q = get_action_serial_V2_1_1(Turn, SLM,QV,Initstates,unavail,lastmove, Forcemove, history, temperature)
        else:
            action, Q = get_action_serial_V2_0_0(Turn, SLM,QV,Initstates,unavail,lastmove, Forcemove, history, temperature)
        
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

def simulate_match_wrapper(args): # dual match
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    p1, p2, models = args
    
    Deck = init_game_3card()
    pair = []

    players = [models[p1],models[p2],models[p2]]
    match_result = simEpisode_notrain([i.clone().detach() for i in Deck], players, 0.0, 0)
    match_result = int(match_result % 3 == 0)
    print(p1, p2, match_result)
    pair.append((p1, p2, match_result))

    players = [models[p2],models[p1],models[p1]]
    match_result = simEpisode_notrain([i.clone().detach() for i in Deck], players, 0.0, 0)
    match_result = int(match_result % 3 == 0)
    print(p2, p1, match_result)
    pair.append((p2, p1, match_result))

    return pair

if __name__ == '__main__':

    eval_device = 'cpu'
    device = 'cpu'

    mp.set_start_method('spawn', force=True)

    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    
    # determine which models participate in the contest
    nsim = 2000
    num_processes = 20


    names = ['H15-V2_1.2-Contester_0035000000','H15-V2_1.1_0010000000']

    models = []
    nplayer = len(names)

    SLM = Network_Pcard_V1_1(20,5)
    QV = Network_Qv_Universal_V1_1(6,15,512)

    SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{names[0]}.pt')))
    QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{names[0]}.pt')))

    SLM.eval()
    QV.eval()

    models.append(
        [SLM,QV]
        )
    
    SLM = Network_Pcard_V1_1(20,5)
    QV = Network_Qv_Universal_V1_1(6,15,256)

    SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{names[1]}.pt')))
    QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{names[1]}.pt')))

    SLM.eval()
    QV.eval()

    models.append(
        [SLM,QV]
        )

    
    ax_player = list(range(nplayer))

    # Create an Elo system instance
    elo_system = EloSystem(num_players=nplayer)

    outfile = os.path.join(wd,'data',f'elo_dual_{names[0]}-{names[-1]}.txt')
    outfile2 = os.path.join(wd,'data',f'wingame_dual_{names[0]}-{names[-1]}.txt')

    try:
        f = open(outfile,'r').readlines()
        elos = [int(s.split()[-1]) for s in f[1:]]
        elo_system.ratings = elos
        csim = int(f[0].split()[1])

        f = open(outfile2,'r').readlines()
        #print([int(s.split('-')[-1].split()[0]) for s in f[1:]])
        wingame = np.array([int(s.split('-')[-1].split()[0]) for s in f[1:]],dtype=np.int64)
        ngames = np.array([int(s.split('/ ')[-1]) for s in f[1:]],dtype=np.int64)
    except:
        csim = 0
        wingame = np.zeros(nplayer,dtype=np.int64)
        ngames = np.zeros(nplayer,dtype=np.int64)
        pass

    print(csim)

    tasks = []
    for _ in range(nsim):
        m1, m2 = random.sample(ax_player, k=2)
        tasks.append((m1, m2, models))

    if num_processes > 1:
    # Create a pool of workers and distribute the tasks
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(simulate_match_wrapper, tasks)
    else:
        # Execute tasks sequentially
        results = []
        for task in tasks:
            result = simulate_match_wrapper(task)
            results.append(result)

    # Update Elo ratings based on the results
    # get match stat as well
    gameout = []
    for pair in results:
        for m1, m2, match_result in pair:
        #print(m1, m2, match_result)
            elo_system.simulate_match(m1, m2, match_result)
            gameout.append((m1,m2,match_result))

    gameout = np.array(gameout)

    #wingame = np.zeros(nplayer,dtype=np.int64)
    #ngames = np.zeros(nplayer,dtype=np.int64)

    for i, p in enumerate(np.unique(gameout[:,0])):
        wingame[i] += (np.sum(gameout[(gameout[:,0] == p),2])+np.sum(1-gameout[(gameout[:,1] == p),2]))
        ngames[i] += (np.sum((gameout[:,0] == p))+np.sum((gameout[:,1] == p)))
    
    # Display the updated ratings
    ratings = elo_system.ratings

    f = open(outfile,'w')
    f.write(f'Nsim {int(csim+nsim)}\n')
    for i in range(nplayer):
        f.write(f'Model {str(names[i]).zfill(3)} - {int(ratings[i])}\n')
    f.close()

    plt.figure(figsize=(10,4))
    plt.title(f'Nsim {int(csim+nsim)}')
    plt.plot(names,ratings)
    plt.savefig(os.path.join(wd,'data',f'elo_dual_{names[0]}-{names[-1]}.png'))


    # wingame stats
    f = open(outfile2,'w')
    f.write(f'Nsim {int(csim+nsim)}\n')
    for i in range(nplayer):
        f.write(f'Model {str(names[i]).zfill(3)} - {wingame[i]} / {ngames[i]}\n')
    f.close()

    plt.figure(figsize=(10,4))
    plt.title(f'Nsim {int(csim+nsim)}')
    plt.plot(names,(wingame/ngames))
    plt.savefig(os.path.join(wd,'data',f'wingame_dual_{names[0]}-{names[-1]}.png'))
    #plt.show()