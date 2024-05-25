import torch
import random
from collections import Counter
from itertools import combinations
import time
import os
import sys
#from model_V2 import *
import numpy as np

from base_utils import *

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
    from model_V2 import *
    from torch.utils.data import TensorDataset, DataLoader, ConcatDataset


    wd = os.path.dirname(__file__)
    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    v_M = 'H15-V5.0_0001000000'
    N_history = int(v_M[1:3])

    SLM = Network_Pcard_V0_0(N_history+4)#,Network_Pcard(N_history+4),Network_Pcard(N_history+4)

    #LM.load_state_dict(torch.load(os.path.join(wd,'models',f'LM_{v_M}.pt')))
    #DM.load_state_dict(torch.load(os.path.join(wd,'models',f'DM_{v_M}.pt')))
    #UM.load_state_dict(torch.load(os.path.join(wd,'models',f'UM_{v_M}.pt')))

    SLM.eval()
    #DM.eval()
    #UM.eval()

    QV = Network_Qv_Universal_V0_0(5)

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