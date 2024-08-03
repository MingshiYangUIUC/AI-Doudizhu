"""

Selfplay base functions. Used by othe scripts, do not run alone.

1. simEpisode_batchpool_softmax: Generate training data by selfplay. Used in 'train_main.py'.
2. gating_batchpool: Contest two models using the same way of #1, only return game result without state and action. Used in 'model_match_fast.py'.

"""

import torch
import random
import os
import numpy as np
import model_utils
import base_utils



#@profile
def simEpisode_batchpool_softmax(Models, temperature, selfplay_device, Nhistory=6, ngame=20, ntask=100, bombchance=0.01, bs_max=30, timestep_limit=9999999999):
    bs_max = bs_max * ngame
    #print('Init wt',Models[0].fc2.weight.data[0].mean())
    #quit()
    if selfplay_device == 'cuda':
        dtypem = torch.float16
        Models[0].to(dtypem).to(selfplay_device)  # SL
        Models[-1].to(dtypem).to(selfplay_device) # QM
    else:
        dtypem = torch.float32
    
    Bz = (Models[-1].scale != 1)

    Index = np.arange(ngame)
    Active = [True for _ in range(ngame)]
    #print(max(Active))

    #Init_states = [init_game_3card() for _ in range(ngame)] # [Lstate, Dstate, Ustate]
    Init_states = [base_utils.init_game_3card_bombmode() if random.choices([True, False], weights=[bombchance, 1-bombchance])[0] else base_utils.init_game_3card() for _ in range(ngame)]
    
    Visible_states = [ist[-1] for ist in Init_states] # the 3 card matrix

    unavail = [torch.zeros(15) for _ in range(ngame)]
    unavail_player = [torch.zeros((3,15)) for _ in range(ngame)] # all cards played by each player

    history = [torch.zeros((Nhistory,15)) for _ in range(ngame)]
    lastmove = [(torch.zeros(15,dtype=torch.float32),(0,0)) for _ in range(ngame)]
    newlast = [() for _ in range(ngame)]

    Turn = torch.zeros(ngame).to(torch.int32)
    Npass = [0 for _ in range(ngame)] # number of pass applying to rules
    Cpass = [0 for _ in range(ngame)] # continuous pass

    Forcemove = [True for _ in range(ngame)] # whether pass is not allowed

    BufferStatesActs = [[[],[],[]] for _ in range(ngame)] # states_actions for 3 players
    BufferRewards = [[0,0,0] for _ in range(ngame)] # rewards for 3 players

    #Full_output = []
    
    LL_organized = []
    F0_organized = []
    F1_organized = []
    LL_reward = []
    F0_reward = []
    F1_reward = []

    stat = np.zeros(3)
    n_steps = 0
    
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

            # use unavail_player instead of CC
            played_cards = unavail_player[_]

            # get actions
            acts = base_utils.avail_actions_cpp_state(lastmove[_][0],lastmove[_][1],playerstate,Forcemove[_])

            # add states and visible to big state
            Bigstate = torch.cat([playerstate.unsqueeze(0),
                                  unavail[_].unsqueeze(0),
                                  #CC,
                                  played_cards, # new feature
                                  visible.unsqueeze(0),
                                  torch.full((1, 15), Tidx), # role
                                  history[_]])
            Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here

            # generate inputs
            # hinput = Bigstate.unsqueeze(0)

            model_inputs.append(Bigstate.unsqueeze(0))
            model_idxs.append(len(acts))
            acts_list.append(acts)

        # use all data to run model
        if selfplay_device == 'cuda':
            torch.cuda.empty_cache()

        model_inputs = torch.concat(model_inputs)

        SL_X.append(model_inputs.clone().detach())
        SL_Y.append(torch.stack(sl_y))
        
        # predict state (SL)
        if selfplay_device == 'cuda':
            model_inter, lstm_out = model(
                model_inputs.to(dtypem).to(selfplay_device)
                )
            model_inter = model_inter.to('cpu', torch.float32)
            lstm_out = lstm_out.to('cpu', torch.float32)
        else:
            model_inter, lstm_out = model(model_inputs)
        #print(model_inter.shape, sl_y.shape)
        
        #role = torch.zeros((model_inter.shape[0],15)) + Tidx

        # use all of model inputs
        model_inter = torch.concat([model_inputs[:,0:8,0].view(model_inputs.size(0), -1), # self
                                    model_inter, # upper and lower states
                                    #role,
                                    lstm_out, # lstm encoded history
                                    ],dim=-1)


        actions_tensors = []
        for acts in acts_list:
            stacked_array = np.stack([a[0] for a in acts]).astype(np.float32)
            actions_tensors.append(torch.from_numpy(stacked_array))

        expanded_model_inter = [mi.unsqueeze(0).expand(actions_tensor.shape[0], -1) for mi, actions_tensor in zip(model_inter, actions_tensors)]
        model_input2 = [torch.cat((expanded_model, actions_tensor), dim=1) for expanded_model, actions_tensor in zip(expanded_model_inter, actions_tensors)]
        '''if Tidx == 1:
            print(lstm_out[0])
            print(model_inputs[0])
            print(model_input2[0][0])
            quit()'''
        # Handle large input (beginning of batchpool) by dividing into chunks
        if selfplay_device == 'cuda':
            model_input2 = torch.cat(model_input2).to(dtypem)
            if len(model_input2) <= bs_max:
                # Original code for small input sizes
                model_output = Models[-1](model_input2.to(selfplay_device)).to('cpu').to(torch.float32).flatten()
            else:
                # Handle large input sizes by processing in batches
                model_output = []
                for i in range(0, len(model_input2), bs_max):
                    batch = model_input2[i:i+bs_max]
                    batch_output = Models[-1](batch.to(selfplay_device)).to('cpu').to(torch.float32).flatten()
                    model_output.append(batch_output)
                model_output = torch.cat(model_output)
        else:
            model_input2 = torch.cat(model_input2)
            model_output = Models[-1](model_input2).flatten()

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
                qa = torch.argmax(output)
                best_act = acts[qa]
            else:
                # get action using probabilistic approach and temperature
                probabilities = torch.softmax(output / temperature, dim=0)
                distribution = torch.distributions.Categorical(probabilities)
                qa = distribution.sample()
                best_act = acts[qa]
            
            '''# add a early terminate to find better samples
            if Turn[_] < 3 and (qa < 0.05 or qa > 0.95) and random.uniform(0,1) > 0.75:
                Active[_] = False
                processed -= 1
                continue
            '''

            #print(torch.argmax(output))
            action = best_act

            if Bz:
                score = base_utils.act2score(action[1])
                BufferRewards[_][Tidx] += score
            
            if Forcemove[_]:
                Forcemove[_] = False

            # conduct a move
            newhiststate = actions_tensors[iout][qa]
            newst = playerstate - newhiststate
            newunavail = unavail[_] + newhiststate
            newhist = torch.roll(history[_],1,dims=0)
            newhist[0] = newhiststate# first row is newest, others are moved downward


            if action[1][0] == 0:
                Cpass[_] += 1
                if Npass[_] < 1:
                    Npass[_] += 1
                else:
                    newlast[_] = (torch.zeros(15,dtype=torch.float32),(0,0))
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

            # add action to unavail_player
            unavail_player[_][Tidx] += newhiststate
            #if _ == 0:
            #    print(Init_states[_][0], unavail_player[_][0], Init_states[_][0]+unavail_player[_][0])

            BufferStatesActs[_][Tidx].append(torch.concat([model_inter[iout].detach(),newhiststate.detach()]).unsqueeze(0))
            #BufferRewards[_][Tidx].append(0)

            Init_states[_][Tidx] = nextstate
            unavail[_] = newunavail
            history[_] = newhist
            lastmove[_] = newlast[_]

            if newst.max() == 0:
                BufferRewards[_][Tidx] += 1 #[torch.as_tensor(BufferRewards[_][Tidx],dtype=torch.float32)+1]
                if Tidx == 1:
                    stat[1] += 1
                    BufferRewards[_][Tidx+1] += 1 #[torch.as_tensor(BufferRewards[_][Tidx+1],dtype=torch.float32)+1]
                    #BufferRewards[_][Tidx-1] += 0 #[torch.as_tensor(BufferRewards[_][Tidx-1],dtype=torch.float32)]
                elif Tidx == 2:
                    stat[2] += 1
                    BufferRewards[_][Tidx-1] += 1 #[torch.as_tensor(BufferRewards[_][Tidx-1],dtype=torch.float32)+1]
                    #BufferRewards[_][Tidx-2] += 0 #[torch.as_tensor(BufferRewards[_][Tidx-2],dtype=torch.float32)]
                elif Tidx == 0:
                    stat[0] += 1
                    #BufferRewards[_][Tidx+1] += 0 #[torch.as_tensor(BufferRewards[_][Tidx+1],dtype=torch.float32)]
                    #BufferRewards[_][Tidx+2] += 0 #[torch.as_tensor(BufferRewards[_][Tidx+2],dtype=torch.float32)]
                
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
                    
                        n_steps += len(p)

                    #SA = [torch.concat(p) for p in BufferStatesActs[_]]
                    #print([sa.shape for sa in SA])
                    #Full_output.append([SA, BufferRewards[_], True])
                #except:
                    #Full_output.append([None,None,False])

        Turn += 1 # all turn += 1

        # break if out of timestep limit
        if n_steps >= timestep_limit:
            break

        # if completed games < ntask, reset state for this index, WHEN NEXT TURN IS LANDLORD
        # if processed < ntask:
        for iout, _ in enumerate(Index):
            while not max(Active) and Turn[_]%3 != 0: # if all games stopped and not landlord turn: just add 1 to turn
                Turn += 1
            if processed < ntask and not Active[_] and Turn[_]%3 == 0: # landlord turn, new game
                # clear buffer
                BufferStatesActs[_] = [[],[],[]]
                BufferRewards[_] = [0,0,0]

                # reset game
                Active[_] = True
                Turn[_] = 0
                #Init_states[_] = init_game_3card()
                Init_states[_] = base_utils.init_game_3card_bombmode() if random.choices([True, False], weights=[bombchance, 1-bombchance])[0] else base_utils.init_game_3card()
                Visible_states[_] = Init_states[_][-1]
                unavail[_] = torch.zeros(15)
                unavail_player[_] = torch.zeros((3,15)) # all cards played by each player
                lastmove[_] = (torch.zeros(15,dtype=torch.float32),(0,0))
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

    Init_states = [base_utils.init_game_3card() for _ in range(ngame)] # [Lstate, Dstate, Ustate]
    Visible_states = [ist[-1] for ist in Init_states] # the 3 card matrix

    unavail = [torch.zeros(15) for _ in range(ngame)]
    unavail_player = [torch.zeros((3,15)) for _ in range(ngame)] # all cards played by each player

    history = [torch.zeros((Nhistory,15)) for _ in range(ngame)]
    lastmove = [(torch.zeros(15,dtype=torch.float32),(0,0)) for _ in range(ngame)]
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

            # use unavail_player instead of CC
            played_cards = unavail_player[_]

            # get actions
            acts = base_utils.avail_actions_cpp_state(lastmove[_][0],lastmove[_][1],playerstate,Forcemove[_])

            # add states and visible to big state
            Bigstate = torch.cat([playerstate.unsqueeze(0),
                                  #str2state_1D(unavail[_]).unsqueeze(0),
                                  unavail[_].unsqueeze(0),
                                  #CC,
                                  played_cards, # new feature
                                  visible.unsqueeze(0), # new feature
                                  torch.full((1, 15), Tidx),
                                  history[_]])
            Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here

            # generate inputs
            model_inputs.append(Bigstate.unsqueeze(0))
            model_idxs.append(len(acts))
            acts_list.append(acts)

        # use all data to run model
        if selfplay_device == 'cuda':
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
        if selfplay_device == 'cuda':
            model_inter, lstm_out = modelS(
                model_inputs.to(dtypem).to(selfplay_device)
                )
            model_inter = model_inter.to('cpu', torch.float32)
            lstm_out = lstm_out.to('cpu', torch.float32)
        else:
            model_inter, lstm_out = modelS(model_inputs)
        #tg1 = time.time()
        #TG += tg1-tg0
        #role = torch.zeros((model_inter.shape[0],15)) + Tidx

        # use all of model inputs
        model_inter = torch.concat([model_inputs[:,0:8,0].view(model_inputs.size(0), -1), # self
                                    model_inter, # upper and lower states
                                    #role,
                                    lstm_out, # lstm encoded history
                                    ],dim=-1)
        model_input2 = []

        actions_tensors = []
        for acts in acts_list:
            stacked_array = np.stack([a[0] for a in acts]).astype(np.float32)
            actions_tensors.append(torch.from_numpy(stacked_array))
        expanded_model_inter = [mi.unsqueeze(0).expand(actions_tensor.shape[0], -1) for mi, actions_tensor in zip(model_inter, actions_tensors)]
        model_input2 = [torch.cat((expanded_model, actions_tensor), dim=1) for expanded_model, actions_tensor in zip(expanded_model_inter, actions_tensors)]
        
        #tg2 = time.time()
        model_output = modelQ(torch.cat(model_input2).to(dtypem).to(selfplay_device)).to('cpu').to(torch.float32).flatten()
        #tg3 = time.time()
        #TG += tg3-tg2
        if selfplay_device == 'cuda':
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
                qa = torch.argmax(output)
                best_act = acts[qa]
            else:
                # get action using probabilistic approach and temperature
                probabilities = torch.softmax(output / temperature, dim=0)
                distribution = torch.distributions.Categorical(probabilities)
                qa = distribution.sample()
                best_act = acts[qa]
            
            #print(torch.argmax(output))
            action = best_act
            
            if Forcemove[_]:
                Forcemove[_] = False

            # conduct a move
            #myst = state2str(playerstate.numpy())
            #cA = Counter(myst)
            #cB = Counter(action[0])
            #newst = ''.join(list((cA - cB).elements()))
            #newhiststate = base_utils.str2state_1D(action[0])
            newhiststate = actions_tensors[iout][qa]
            newst = playerstate - newhiststate
            newunavail = unavail[_] + newhiststate
            newhist = torch.roll(history[_],1,dims=0)
            #newhiststate = str2state_1D(action[0])# str2state(action[0]).sum(axis=-2,keepdims=True) 
            
            newhist[0] = newhiststate# first row is newest, others are moved downward
            
            unavail_player[_][Tidx] += newhiststate

            if action[1][0] == 0:
                Cpass[_] += 1
                if Npass[_] < 1:
                    Npass[_] += 1
                else:
                    newlast[_] = (torch.zeros(15,dtype=torch.float32),(0,0))
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
                Init_states[_] = base_utils.init_game_3card()
                Visible_states[_] = Init_states[_][-1]
                #unavail[_] = ''
                unavail[_] = torch.zeros(15)
                unavail_player[_] = torch.zeros((3,15)) # all cards played by each player
                lastmove[_] = (torch.zeros(15,dtype=torch.float32),(0,0))
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
    #from model_utils import *
    from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

    wd = os.path.dirname(__file__)
    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)


    #SLM = Network_Pcard_V2_1_Trans(15+7, 7, y=1, x=15, trans_heads=4, trans_layers=6, hiddensize=512)
    SLM = model_utils.Network_Pcard_V2_2_BN_dropout(15+7, 7, y=1, x=15, lstmsize=512, hiddensize=512, dropout_rate = 0.2)
    QV = model_utils.Network_Qv_Universal_V1_2_BN_dropout(input_size=11*15,lstmsize=512, hsize=512, dropout_rate = 0.2, scale_factor=1.2,offset_factor=0.0)
    #SLM.load_state_dict(torch.load(os.path.join(wd,'models','SLM_H15-VBx5_128-128-128_0.01_0.0001-0.0001_256_0000000000.pt')))
    #QV.load_state_dict(torch.load(os.path.join(wd,'models','QV_H15-VBx5_128-128-128_0.01_0.0001-0.0001_256_0000000000.pt')))
    #quit()
    #SLM.load_state_dict(torch.load(os.path.join(wd,'models','SLM_H15-V2_2.3_0103300000.pt')))
    #QV.load_state_dict(torch.load(os.path.join(wd,'models','QV_H15-V2_2.3_0103300000.pt')))
    SLM.eval()
    QV.eval()

    #torch.save(SLM.state_dict(),os.path.join(wd,'test_models',f'SLM_Trans.pt'))
    #torch.save(QV.state_dict(),os.path.join(wd,'models',f'QV_Trans.pt'))

    N_episodes = 128
    ng = 32
    
    seed = random.randint(0,1000000000)
    seed = 12333
    print(seed)
    random.seed(seed)

    #SLM = Network_Pcard_V2_1(22,7,1,15, 512,512)
    #QV = Network_Qv_Universal_V1_1(6,15,512)
    with torch.no_grad():
        with torch.inference_mode():
            out = simEpisode_batchpool_softmax([SLM,QV], 0, 'cpu', Nhistory=15, ngame=ng, ntask=N_episodes,bombchance=0.0,
                                            bs_max=64)
            print(out[-1])
            from matplotlib import pyplot as plt
            #print(list(out[1][0].numpy()))
            plt.hist(out[1][1].numpy(),bins=100)
            plt.show()

    max_memory_used = torch.cuda.max_memory_allocated('cuda')
    #print(f"Maximum memory used by the model: {max_memory_used / (1024 ** 2)} MB")

    # You can also get a detailed memory summary
    #print(torch.cuda.memory_summary('cuda'))
    
    '''gatingresult = gating_batchpool(models, 0, 'cuda', Nhistory=15, ngame=ng, ntask=N_episodes, rseed=seed)
    print('')
    LW = (np.array(gatingresult)==0).sum()

    gatingresult = gating_batchpool(models[::-1], 0, 'cuda', Nhistory=15, ngame=ng, ntask=N_episodes, rseed=seed)
    print('')
    FW = (np.array(gatingresult)!=0).sum()

    print(LW, FW, LW+FW, N_episodes*2, round((LW+FW)/N_episodes/2*100,1))'''