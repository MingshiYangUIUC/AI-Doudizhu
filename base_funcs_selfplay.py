import torch
import random
from collections import Counter
from itertools import combinations
import time
import os
import sys
from model_utils import *
import numpy as np

from base_utils import *




#@profile
def simEpisode_batchpool_softmax(Models, temperature, selfplay_device, Nhistory=6, ngame=20, ntask=100, bombchance=0.01):
    #print('Init wt',Models[0].fc2.weight.data[0].mean())
    #quit()
    if selfplay_device == 'cuda':
        dtypem = torch.float16
        Models[0].to(dtypem).to(selfplay_device)  # SL
        Models[-1].to(dtypem).to(selfplay_device) # QM
    else:
        dtypem = torch.float32

    Index = np.arange(ngame)
    Active = [True for _ in range(ngame)]
    #print(max(Active))

    #Init_states = [init_game_3card() for _ in range(ngame)] # [Lstate, Dstate, Ustate]
    Init_states = [init_game_3card_bombmode() if random.choices([True, False], weights=[bombchance, 1-bombchance])[0] else init_game_3card() for _ in range(ngame)]
    
    Visible_states = [ist[-1] for ist in Init_states] # the 3 card matrix

    unavail = [torch.zeros(15) for _ in range(ngame)]
    history = [torch.zeros((Nhistory,15)) for _ in range(ngame)]
    lastmove = [('',(0,0)) for _ in range(ngame)]
    newlast = [() for _ in range(ngame)]

    Turn = torch.zeros(ngame).to(torch.int32)
    Npass = [0 for _ in range(ngame)] # number of pass applying to rules
    Cpass = [0 for _ in range(ngame)] # continuous pass

    Forcemove = [True for _ in range(ngame)] # whether pass is not allowed

    BufferStatesActs = [[[],[],[]] for _ in range(ngame)] # states_actions for 3 players
    BufferRewards = [[-1,-1,-1] for _ in range(ngame)] # rewards for 3 players

    #Full_output = []
    
    LL_organized = []
    F0_organized = []
    F1_organized = []
    LL_reward = []
    F0_reward = []
    F1_reward = []

    stat = np.zeros(3)
    
    processed = ngame

    #Games = ['' for _ in range(ngame)]
    SL_X = [] # no order, ok to use one list
    SL_Y = []

    while processed < ntask or True in Active: # game loop
        # serially get all actions and states

        model_inputs = []
        model_idxs = []
        acts_list = []
        sl_y = []
        
        model = Models[0]

        Tidx = (Turn[0]%3).item()

        for iin, _ in enumerate(Index[Active]):

            #Tidx = Turn[_]%3
            #print(Tidx)
            #playerstate, model = Init_states[_][Tidx], Models[Tidx] # Same model should be used for all Active
            playerstate = Init_states[_][Tidx]
            upper_state = Init_states[_][(Tidx-1)%3]#.sum(dim=0)
            lower_state = Init_states[_][(Tidx+1)%3]#.sum(dim=0)
            sl_y.append(torch.cat((upper_state,lower_state)))
            #print(Turn)
            visible = Visible_states[_] # 3 cards from landlord, fixed for one full game

            # get card count
            card_count = [int(p.sum()) for p in Init_states[_][:-1]]
            CC = torch.zeros((3,15))
            CC[0][:min(card_count[0],15)] = 1
            CC[1][:min(card_count[1],15)] = 1
            CC[2][:min(card_count[2],15)] = 1

            # get actions
            #print('a')
            #print(Forcemove[_])
            

            acts = avail_actions_cpp(lastmove[_][0],lastmove[_][1],playerstate,Forcemove[_])

            # add states and visible to big state

            Bigstate = torch.cat([playerstate.unsqueeze(0),
                                  #str2state_1D(unavail[_]).unsqueeze(0),
                                  unavail[_].unsqueeze(0),
                                  CC,
                                  visible.unsqueeze(0), # new feature
                                  torch.full((1, 15), Tidx),
                                  history[_]])
            Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here

            # generate inputs
            # hinput = Bigstate.unsqueeze(0)

            model_inputs.append(Bigstate.unsqueeze(0))
            model_idxs.append(len(acts))
            acts_list.append(acts)


        # use all data to run model
        torch.cuda.empty_cache()

        model_inputs = torch.concat(model_inputs)

        #print(model_inputs.shape)
        #quit()
        SL_X.append(model_inputs.clone().detach())
        SL_Y.append(torch.stack(sl_y))
        # predict state (SL)
        model_inter = model(
            model_inputs.to(dtypem).to(selfplay_device)
            ).to('cpu').to(torch.float32)
        #print(model_inter.shape)
        #quit()
        #print(model_inter.shape,torch.stack(sl_y).shape)

        role = torch.zeros((model_inter.shape[0],15)) + Tidx

        model_inter = torch.concat([model_inputs[:,0,0], # self
                                    model_inputs[:,7,0], # history
                                    model_inter, # upper and lower states
                                    role],dim=-1)
        model_input2 = []

        for i, mi in enumerate(model_inter):
            actions_tensor = torch.stack([str2state_compressed_1D(a[0]) for a in acts_list[i]])
            mi_expanded = mi.unsqueeze(0).expand(actions_tensor.shape[0],-1)  # Expand mi to match the batch size of actions_tensor
            input_i = torch.cat((mi_expanded, actions_tensor), dim=1)
            model_input2.append(input_i)

        #print(model_inter.shape)
        #quit()
        model_output = Models[-1](torch.cat(model_input2).to(dtypem).to(selfplay_device)).to('cpu').to(torch.float32).flatten()


        # conduct actions for all instances
        for iout, _ in enumerate(Index[Active]):
            
            playerstate = Init_states[_][Tidx]
            #Bigstate = bstates_list[iout]

            idx_start = sum(model_idxs[:iout])
            idx_end = sum(model_idxs[:iout+1])

            # get q values
            output = model_output[idx_start:idx_end]#.clone().detach()
            acts = acts_list[iout]

            if temperature == 0:
                #q = torch.max(output)
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
            #myst = state2str(playerstate.numpy())
            #cA = Counter(myst)
            #cB = Counter(action[0])
            #newst = ''.join(list((cA - cB).elements()))
            newhiststate = str2state_1D(action[0])
            newst = playerstate - newhiststate
            newunavail = unavail[_] + newhiststate
            newhist = torch.roll(history[_],1,dims=0)
            #newhiststate = str2state_1D(action[0])# str2state(action[0]).sum(axis=-2,keepdims=True) 
            
            newhist[0] = newhiststate# first row is newest, others are moved downward


            if action[1][0] == 0:
                Cpass[_] += 1
                if Npass[_] < 1:
                    Npass[_] += 1
                else:
                    newlast[_] = ('',(0,0))
                    Npass[_] = 0
                    Forcemove[_] = True
            else:
                newlast[_] = action
                Npass[_] = 0
                Cpass[_] = 0

            # record and update
            #nextstate = str2state_1D(newst)
            nextstate = newst
            #print(newst)

            BufferStatesActs[_][Tidx].append(torch.concat([model_inter[iout].detach(),newhiststate.detach()]).unsqueeze(0))
            #BufferRewards[_][Tidx].append(0)

            Init_states[_][Tidx] = nextstate
            unavail[_] = newunavail
            history[_] = newhist
            lastmove[_] = newlast[_]

            if newst.max() == 0:
                BufferRewards[_][Tidx] = 1 #[torch.as_tensor(BufferRewards[_][Tidx],dtype=torch.float32)+1]
                if Tidx == 1:
                    stat[1] += 1
                    BufferRewards[_][Tidx+1] = 1 #[torch.as_tensor(BufferRewards[_][Tidx+1],dtype=torch.float32)+1]
                    BufferRewards[_][Tidx-1] = 0 #[torch.as_tensor(BufferRewards[_][Tidx-1],dtype=torch.float32)]
                elif Tidx == 2:
                    stat[2] += 1
                    BufferRewards[_][Tidx-1] = 1 #[torch.as_tensor(BufferRewards[_][Tidx-1],dtype=torch.float32)+1]
                    BufferRewards[_][Tidx-2] = 0 #[torch.as_tensor(BufferRewards[_][Tidx-2],dtype=torch.float32)]
                elif Tidx == 0:
                    stat[0] += 1
                    BufferRewards[_][Tidx+1] = 0 #[torch.as_tensor(BufferRewards[_][Tidx+1],dtype=torch.float32)]
                    BufferRewards[_][Tidx+2] = 0 #[torch.as_tensor(BufferRewards[_][Tidx+2],dtype=torch.float32)]
                
                Active[_] = False

                # send data to output collection
                #try:
                for i,p in enumerate(BufferStatesActs[_]):
                    if len(p) > 0:
                        if i == 0:
                            LL_organized.append(torch.concat(p))
                            LL_reward.append(torch.zeros(len(p))+BufferRewards[_][i])
                        elif i == 1:
                            F0_organized.append(torch.concat(p))
                            F0_reward.append(torch.zeros(len(p))+BufferRewards[_][i])
                        elif i == 2:
                            F1_organized.append(torch.concat(p))
                            F1_reward.append(torch.zeros(len(p))+BufferRewards[_][i])
                    
                    #SA = [torch.concat(p) for p in BufferStatesActs[_]]
                    #print([sa.shape for sa in SA])
                    #Full_output.append([SA, BufferRewards[_], True])
                #except:
                    #Full_output.append([None,None,False])

        Turn += 1 # all turn += 1

        # if completed games < ntask, reset state for this index, WHEN NEXT TURN IS LANDLORD
        # if processed < ntask:
        for iout, _ in enumerate(Index):
            while not max(Active) and Turn[_]%3 != 0: # if all games stopped and not landlord turn: just add 1 to turn
                Turn += 1
            if processed < ntask and not Active[_] and Turn[_]%3 == 0: # landlord turn, new game
                # clear buffer
                BufferStatesActs[_] = [[],[],[]]
                BufferRewards[_] = [-1,-1,-1]

                # reset game
                Active[_] = True
                Turn[_] = 0
                #Init_states[_] = init_game_3card()
                Init_states[_] = init_game_3card_bombmode() if random.choices([True, False], weights=[bombchance, 1-bombchance])[0] else init_game_3card()
                Visible_states[_] = Init_states[_][-1]
                unavail[_] = torch.zeros(15)
                lastmove[_] = ('',(0,0))
                newlast[_] = ()
                Npass[_] = 0
                Cpass[_] = 0
                Forcemove[_] = True
                processed += 1
                print(str(processed).zfill(5),'/', str(ntask).zfill(5), '   ',end='\r')
        #print(Active)
        #print(processed, ntask, Turn)
    
    LL_organized = torch.cat(LL_organized)
    F0_organized = torch.cat(F0_organized)
    F1_organized = torch.cat(F1_organized)
    LL_reward = torch.cat(LL_reward)
    F0_reward = torch.cat(F0_reward)
    F1_reward = torch.cat(F1_reward)
    #print(LL_organized.shape,F0_organized.shape,F1_organized.shape)
    #print(LL_reward.shape)
    SL_X = torch.cat(SL_X)
    SL_Y = torch.cat(SL_Y)
    
    return [LL_organized,F0_organized,F1_organized], [LL_reward,F0_reward,F1_reward], SL_X, SL_Y, stat
    #return Full_output, SL_X, SL_Y

def gating_batchpool(Models, temperature, selfplay_device, Nhistory=6, ngame=20, ntask=100, rseed=0):
    # TG, TC = 0, time.time()
    # similar to above func but does not record training data
    # use two sets of models, first set plays as landlord
    random.seed(rseed)

    if selfplay_device == 'cuda':
        dtypem = torch.float16
        Models[0][0].to(dtypem).to(selfplay_device)  # SL for p1
        Models[0][1].to(dtypem).to(selfplay_device) # QM for p1
        Models[1][0].to(dtypem).to(selfplay_device)  # SL for p2
        Models[1][1].to(dtypem).to(selfplay_device) # QM for p2
    else:
        dtypem = torch.float32

    Index = torch.arange(ngame)
    Active = [True for _ in range(ngame)]

    Init_states = [init_game_3card() for _ in range(ngame)] # [Lstate, Dstate, Ustate]
    Visible_states = [ist[-1] for ist in Init_states] # the 3 card matrix

    unavail = [torch.zeros(15) for _ in range(ngame)]
    history = [torch.zeros((Nhistory,15)) for _ in range(ngame)]
    lastmove = [('',(0,0)) for _ in range(ngame)]
    newlast = [() for _ in range(ngame)]

    Turn = torch.zeros(ngame).to(torch.int32)
    Npass = [0 for _ in range(ngame)] # number of pass applying to rules
    Cpass = [0 for _ in range(ngame)] # continuous pass

    Forcemove = [True for _ in range(ngame)] # whether pass is not allowed
    
    processed = ngame

    result = []

    while processed < ntask or True in Active: # game loop
        # serially get all actions and states

        model_inputs = []
        model_idxs = []
        acts_list = []
        
        

        for iin, _ in enumerate(Index[Active]):

            Tidx = Turn[_]%3
            #playerstate, model = Init_states[_][Tidx], Models[Tidx] # Same model should be used for all Active
            playerstate = Init_states[_][Tidx]

            #print(Turn)
            visible = Visible_states[_] # 3 cards from landlord, fixed for one full game

            # get card count
            card_count = [int(p.sum()) for p in Init_states[_][:-1]]
            CC = torch.zeros((3,15))
            CC[0][:min(card_count[0],15)] = 1
            CC[1][:min(card_count[1],15)] = 1
            CC[2][:min(card_count[2],15)] = 1

            # get actions
            acts = avail_actions_cpp(lastmove[_][0],lastmove[_][1],playerstate,Forcemove[_])

            # add states and visible to big state
            Bigstate = torch.cat([playerstate.unsqueeze(0),
                                  #str2state_1D(unavail[_]).unsqueeze(0),
                                  unavail[_].unsqueeze(0),
                                  CC,
                                  visible.unsqueeze(0), # new feature
                                  torch.full((1, 15), Tidx),
                                  history[_]])
            Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here

            # generate inputs
            model_inputs.append(Bigstate.unsqueeze(0))
            model_idxs.append(len(acts))
            acts_list.append(acts)

        # use all data to run model
        torch.cuda.empty_cache()

        if Tidx == 0: # first set of model
            modelS = Models[0][0]
            modelQ = Models[0][1]
        else:
            modelS = Models[1][0]
            modelQ = Models[1][1]

        model_inputs = torch.concat(model_inputs)

        # predict state (SL)
        #tg0 = time.time()
        model_inter = modelS(
            model_inputs.to(dtypem).to(selfplay_device)
            ).to('cpu').to(torch.float32)
        #tg1 = time.time()
        #TG += tg1-tg0
        role = torch.zeros((model_inter.shape[0],15)) + Tidx

        model_inter = torch.concat([model_inputs[:,0,0], # self
                                    model_inputs[:,7,0], # history
                                    model_inter, # upper and lower states
                                    role],dim=-1)
        model_input2 = []

        for i, mi in enumerate(model_inter):
            input_i = torch.stack([torch.cat((mi,str2state_1D(a[0]))) for a in acts_list[i]])
            model_input2.append(input_i)
        #tg2 = time.time()
        model_output = modelQ(torch.cat(model_input2).to(dtypem).to(selfplay_device)).to('cpu').to(torch.float32).flatten()
        #tg3 = time.time()
        #TG += tg3-tg2
        torch.cuda.empty_cache()

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
            #myst = state2str(playerstate.numpy())
            #cA = Counter(myst)
            #cB = Counter(action[0])
            #newst = ''.join(list((cA - cB).elements()))
            newhiststate = str2state_1D(action[0])
            newst = playerstate - newhiststate
            newunavail = unavail[_] + newhiststate
            newhist = torch.roll(history[_],1,dims=0)
            #newhiststate = str2state_1D(action[0])# str2state(action[0]).sum(axis=-2,keepdims=True) 
            
            newhist[0] = newhiststate# first row is newest, others are moved downward

            if action[1][0] == 0:
                Cpass[_] += 1
                if Npass[_] < 1:
                    Npass[_] += 1
                else:
                    newlast[_] = ('',(0,0))
                    Npass[_] = 0
                    Forcemove[_] = True
            else:
                newlast[_] = action
                Npass[_] = 0
                Cpass[_] = 0

            # record and update
            #nextstate = str2state_1D(newst)
            nextstate = newst
            #print(newst)

            Init_states[_][Tidx] = nextstate
            unavail[_] = newunavail
            history[_] = newhist
            lastmove[_] = newlast[_]

            if newst.max() == 0:
                
                Active[_] = False

                result.append(Tidx.item()) # append winners

        Turn += 1 # all turn += 1

        # if completed games < ntask, reset state for this index, WHEN NEXT TURN IS LANDLORD
        # if processed < ntask:
        for iout, _ in enumerate(Index):
            while not max(Active) and Turn[_]%3 != 0: # if all games stopped and not landlord turn: just add 1 to turn
                Turn += 1
            if processed < ntask and not Active[_] and Turn[_]%3 == 0: # landlord turn, new game
                # clear buffer

                # reset game
                Active[_] = True
                Turn[_] = 0
                Init_states[_] = init_game_3card()
                Visible_states[_] = Init_states[_][-1]
                #unavail[_] = ''
                unavail[_] = torch.zeros(15)
                lastmove[_] = ('',(0,0))
                newlast[_] = ()
                Npass[_] = 0
                Cpass[_] = 0
                Forcemove[_] = True
                processed += 1
                print(str(processed).zfill(5),'/', str(ntask).zfill(5), '   ',end='\r')
        #print(Active)
        #print(processed, ntask, Turn)
    #TC = time.time()-TC
    #print(TG,TC,TG/TC)
    return np.array(result)

if __name__ == '__main__':
    from model_utils import *
    from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

    wd = os.path.dirname(__file__)
    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


    '''names = ['H15-V2_2.2_0020000000','H15-V2_2.2_0020000000']
    models = []
    for name in names:
        SLM = Network_Pcard_V2_1(15+7, 7, y=1, x=15, lstmsize=512, hiddensize=1024)
        QV = Network_Qv_Universal_V1_1(6,15,1024)

        SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{name}.pt')))
        QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{name}.pt')))

        SLM.eval()
        QV.eval()

        models.append(
            [SLM,QV]
            )'''
    SLM = Network_Pcard_V2_1_Trans(15+7, 7, y=1, x=15, trans_heads=4, trans_layers=6, hiddensize=512)
    #SLM = Network_Pcard_V2_1(15+7, 7, y=1, x=15, lstmsize=512, hiddensize=512)
    QV = Network_Qv_Universal_V1_1(6,15,512)
    SLM.eval()
    QV.eval()

    torch.save(SLM.state_dict(),os.path.join(wd,'test_models',f'SLM_Trans.pt'))
    #torch.save(QV.state_dict(),os.path.join(wd,'models',f'QV_Trans.pt'))

    N_episodes = 512
    ng = 64
    
    seed = random.randint(0,1000000000)
    seed = 12333
    print(seed)
    random.seed(seed)

    #SLM = Network_Pcard_V2_1(22,7,1,15, 512,512)
    #QV = Network_Qv_Universal_V1_1(6,15,512)
    with torch.no_grad():
        out = simEpisode_batchpool_softmax([SLM,QV], 0, 'cuda', Nhistory=15, ngame=ng, ntask=N_episodes)
        print(out[-1])
    
    '''gatingresult = gating_batchpool(models, 0, 'cuda', Nhistory=15, ngame=ng, ntask=N_episodes, rseed=seed)
    print('')
    LW = (np.array(gatingresult)==0).sum()

    gatingresult = gating_batchpool(models[::-1], 0, 'cuda', Nhistory=15, ngame=ng, ntask=N_episodes, rseed=seed)
    print('')
    FW = (np.array(gatingresult)!=0).sum()

    print(LW, FW, LW+FW, N_episodes*2, round((LW+FW)/N_episodes/2*100,1))'''