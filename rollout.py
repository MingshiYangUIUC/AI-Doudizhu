from model_utils import *
from base_utils import *

def resample_state(Turn, Initstates, unavail, model_inter):
    cprob = model_inter.detach().numpy().round(1).reshape(2,15)

    cprob[0] = cprob[0] / np.sum(cprob[0])
    cprob[1] = cprob[1] / np.sum(cprob[1])
    
    # total count is exact values
    total_count = Initstates[Turn].sum(axis=-2,keepdims=True).detach().numpy().flatten()
    total_count += str2state(unavail).sum(axis=-2,keepdims=True).numpy().flatten()
    #print(Turn, total_count)
    total_count[:13] = 4 - total_count[:13]
    total_count[13:] = 1 - total_count[13:]
    total_count = np.int32(total_count)

    sample1 = np.zeros(15, dtype=int)
    sample2 = np.zeros(15, dtype=int)

    # Sample cards for each player separately but ensuring total count matches
    idx = np.arange(15)
    np.random.shuffle(idx)
    ncard1 = 0
    max1 = Initstates[(Turn-1)%3].sum()

    for _ in range(15):  # For each card type
        i = idx[_]
        # Allocate cards based on total counts for each type
        total = total_count[i]
        if total == 0:
            continue
        count1 = np.random.binomial(total, cprob[0][i] / (cprob[0][i] + cprob[1][i]))
        if ncard1 + count1 > max1:
            count1 = max1 - ncard1
            sample1[i] = count1
            ncard1 += count1
            break
        else:
            sample1[i] = count1
            ncard1 += count1

    sample2 = total_count - sample1

    # corrective measure if sample2 has total > its true
    diff2 = int(sample2.sum() - Initstates[(Turn+1)%3].sum())
    if diff2 > 0: # remove several one based on probability
        for _ in range(diff2):
            #probabilities = sample2 / sample2.sum()
            prob = np.ones(15)
            prob[sample2==0] = 0
            prob /= prob.sum()
            chosen_card = np.random.choice(15, p=prob)
            sample1[chosen_card] += 1
            sample2[chosen_card] -= 1
    #print(sample2)
    return sample1, sample2

#@profile
def rollout_2_2_2(Turn, SLM, QV, sample_states, overwrite_action, unavail, lastmove, Forcemove, history, temperature, Npass, Cpass, depth=3):

    rTurn = Turn
    newlast = lastmove

    d = 0
    end = False

    SLM.to('cuda')
    QV.to('cuda')

    while True and d <= depth:

        player = sample_states[rTurn%3]

        if d > 0: # rollout

            #player = sample_states[rTurn%3]#.clone().detach()
            visible = sample_states[-1]#.clone().detach()

            # get card count
            card_count = [int(p.sum()) for p in sample_states]
            CC = torch.zeros((3,15))
            CC[0][:min(card_count[0],15)] = 1
            CC[1][:min(card_count[1],15)] = 1
            CC[2][:min(card_count[2],15)] = 1

            # get action
            Bigstate = torch.cat([player.sum(axis=-2,keepdims=True).unsqueeze(0),
                                    str2state(unavail).sum(axis=-2,keepdims=True).unsqueeze(0),
                                    CC.unsqueeze(1),
                                    visible.sum(axis=-2,keepdims=True).unsqueeze(0), # new feature
                                    torch.zeros((1,15)).unsqueeze(0) + rTurn%3, # role feature
                                    history.sum(axis=-2,keepdims=True)])
            hinput = Bigstate.unsqueeze(0)
            model_inter = SLM(hinput.to('cuda')).to('cpu')
            role = torch.zeros((model_inter.shape[0],15)) + rTurn%3
            acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)
            model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                        hinput[:,7].sum(dim=-2),
                                        model_inter,
                                        role],dim=-1)
            model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])
            # get q values
            output = QV(model_input2.to('cuda')).to('cpu').flatten()

            if temperature == 0:
                Q = torch.max(output).item()
                best_act = acts[torch.argmax(output)]
            else:
                # get action using probabilistic approach and temperature
                probabilities = torch.softmax(output / temperature, dim=0)
                distribution = torch.distributions.Categorical(probabilities)
                
                q = distribution.sample()
                best_act = acts[q]
                Q = output[q].item()
            
            action = best_act

        else: # use action
            action = overwrite_action

        if Forcemove:
            Forcemove = False

        # conduct a move
        myst = state2str(player.sum(dim=0).numpy())
        cA = Counter(myst)
        cB = Counter(action[0])
        newst = ''.join(list((cA - cB).elements()))
        newunavail = unavail + action[0]
        newhist = torch.roll(history,1,dims=0)
        
        newhist[0] = str2state(action[0]).sum(axis=-2,keepdims=True) # first row is newest, others are moved downward

        play = action[0]
        if action[1][0] == 0:
            play = 'pass'
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
        
        # update
        nextstate = str2state(newst)
        sample_states[rTurn%3] = nextstate
        unavail = newunavail
        #print(newlast)
        history = newhist
        lastmove = newlast
        
        #print(Q)

        if len(newst) == 0:
            end = True
            break

        rTurn += 1
        d += 1

    #W = 0
    if end:
        Q = 0.0
        if rTurn%3 == 0 and Turn%3 == 0:
            Q = 1.0
        if rTurn%3 != 0 and Turn%3 != 0:
            Q = 1.0
    
    SLM.to('cpu')
    QV.to('cpu')

    return Q

def get_action_adv(Turn, SLM, QV, Initstates, unavail, lastmove, Forcemove, history, temperature, Npass, Cpass, nAct=5, nRoll=10, ndepth=3, maxtime=999999, sleep=True):
    
    t0 = time.time()

    player = Initstates[Turn%3]#.clone().detach()
    visible = Initstates[-1]#.clone().detach()

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((3,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.sum(axis=-2,keepdims=True).unsqueeze(0),
                            str2state(unavail).sum(axis=-2,keepdims=True).unsqueeze(0),
                            CC.unsqueeze(1),
                            visible.sum(axis=-2,keepdims=True).unsqueeze(0), # new feature
                            torch.zeros((1,15)).unsqueeze(0) + Turn%3, # role feature
                            history.sum(axis=-2,keepdims=True)])
    #print(Bigstate)
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    # get all actions

    acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)

    # generate inputs 2
    model_inter2 = torch.concat([hinput[:,0].sum(dim=-2),
                                hinput[:,7].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter2.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()

    # get N best actions to sample from!
    N = min(nAct,len(acts))

    if temperature == 0:
        top_n_indices = torch.topk(output, N).indices
        n_actions = [acts[idx] for idx in top_n_indices]
        n_Q_values = output[top_n_indices]
    else:
        probabilities = torch.softmax(output / temperature, dim=0)
        distribution = torch.distributions.Categorical(probabilities)
        sampled_indices = distribution.sample((N,))
        n_actions = [acts[idx] for idx in sampled_indices]
        n_Q_values = output[sampled_indices]

    new_Q = torch.zeros(N)
    r_count = 0
    if N == 1:
        new_Q = n_Q_values
        if sleep:
            time.sleep(maxtime)
    else:
        #print(r_count, nRoll, time.time(),t0,maxtime)
        while r_count < nRoll and time.time() - t0 < maxtime:
            Qroll = []
            for i in range(N) : # resample given action
                action = n_actions[i]
                # original value has some weight (good if simulation number is small)

                # construct fake initsates, and rollout
                sample1, sample2 = resample_state(Turn%3, Initstates, unavail, model_inter)
                sample_states = [None,None,None,Initstates[-1].clone()]
                sample_states[Turn%3] = Initstates[Turn%3].clone()
                sample_states[(Turn-1)%3] = str2state(''.join([r2c_base[i]*sample1[i] for i in range(15)]))
                sample_states[(Turn+1)%3] = str2state(''.join([r2c_base[i]*sample2[i] for i in range(15)]))
                Qroll.append(
                    rollout_2_2_2(Turn, SLM, QV, sample_states, action.copy(), unavail, lastmove.copy(), Forcemove, history.clone(), temperature, Npass, Cpass,
                                depth=ndepth))

            Qroll = torch.as_tensor(Qroll)

            new_Q += Qroll

            r_count += 1
        new_Q /= r_count
    
        print(f'Scanned {N} actions, {r_count} bootstraps, depth {ndepth}.')

    best_action = n_actions[torch.argmax(new_Q)]
    Q = torch.max(new_Q)
    #print(n_Q_values.numpy().round(2), new_Q.numpy().round(2))
    #print(Turn, n_actions[0],best_action)
    return best_action, Q

#@profile
def rollout_batch(Turn, SLM, QV, sample_states, overwrite_action, unavail, lastmove, Forcemove, history, temperature, Npass, Cpass, depth=3, selfplay_device='cuda'):
    
    # do all sample states in a batch
    # sample_states are different (passed here)
    # Turn is the same (passed here)
    # overwrite_actions are same or different (passed here)
    # unavail, lastmove, Forcemove, history, Npass, Cpass, are the same initially but will be different 
    
    # set up batch except (passed_here)
    ngame = len(sample_states)

    Visible_states = [ist[-1] for ist in sample_states] # the 3 card matrix

    unavail = [unavail for _ in range(ngame)]
    history = [torch.zeros((15,1,15)) for _ in range(ngame)]
    lastmove = [lastmove.copy() for _ in range(ngame)]
    newlast = [lastmove.copy() for _ in range(ngame)]

    Npass = [Npass for _ in range(ngame)] # number of pass applying to rules
    Cpass = [Cpass for _ in range(ngame)] # continuous pass

    Forcemove = [Forcemove for _ in range(ngame)] # whether pass is not allowed

    rTurn = Turn

    d = 0

    Index = torch.arange(ngame)
    Active = [True for _ in range(ngame)]


    results = torch.zeros(ngame)

    while d <= depth and True in Active:

        if d > 0: # batch rollout

            model_inputs = []
            model_idxs = []
            acts_list = []
            
            # process model input

            for iin, _ in enumerate(Index[Active]):
                Tidx = rTurn%3
                playerstate = sample_states[_][Tidx]

                visible = Visible_states[_] # 3 cards from landlord, fixed for one full game

                # get card count
                card_count = [int(p.sum()) for p in sample_states[_]]
                CC = torch.zeros((3,15))
                CC[0][:min(card_count[0],15)] = 1
                CC[1][:min(card_count[1],15)] = 1
                CC[2][:min(card_count[2],15)] = 1

                # get actions
                acts = avail_actions(lastmove[_][0],lastmove[_][1],playerstate,Forcemove[_])
                #print(acts,lastmove[_][0])

                # add states and visible to big state
                Bigstate = torch.cat([playerstate.sum(axis=-2,keepdims=True).unsqueeze(0),
                                    #str2state(unavail[_]).sum(axis=-2,keepdims=True).unsqueeze(0),
                                    str2state_compressed(unavail[_]).unsqueeze(0),
                                    CC.unsqueeze(1),
                                    visible.sum(axis=-2,keepdims=True).unsqueeze(0), # new feature
                                    torch.zeros((1,15)).unsqueeze(0) + Tidx, # role feature
                                    history[_]])

                # generate inputs
                model_inputs.append(Bigstate.unsqueeze(0))
                model_idxs.append(len(acts))
                acts_list.append(acts)
            
            model_inputs = torch.concat(model_inputs)

            # predict state (SL)
            model_inter = SLM(
                model_inputs.to(selfplay_device)
                ).to('cpu')

            role = torch.zeros((model_inter.shape[0],15)) + Tidx

            model_inter = torch.concat([model_inputs[:,0].sum(dim=-2), # self
                                        model_inputs[:,7].sum(dim=-2), # history
                                        model_inter, # upper and lower states
                                        role],dim=-1)
            model_input2 = []

            for i, mi in enumerate(model_inter):
                actions_tensor = torch.stack([str2state_compressed_1D(a[0]) for a in acts_list[i]])
                mi_expanded = mi.unsqueeze(0).expand(actions_tensor.shape[0],-1)  # Expand mi to match the batch size of actions_tensor
                input_i = torch.cat((mi_expanded, actions_tensor), dim=1)
                model_input2.append(input_i)

            #print(model_input2)
            

            model_output = QV(torch.cat(model_input2).to(selfplay_device)).to('cpu').flatten()
            
            #print(model_output)
            #quit()
            # get action list
            selected_actions = []
            selected_qs = []

            for iout, _ in enumerate(Index[Active]):
                
                playerstate = sample_states[_][Tidx]
                #Bigstate = bstates_list[iout]

                idx_start = sum(model_idxs[:iout])
                idx_end = sum(model_idxs[:iout+1])

                # get q values
                output = model_output[idx_start:idx_end]#.clone().detach()
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
                selected_actions.append(action)
                #print(q)
                selected_qs.append(q)
        
        else: # use action list
            selected_actions = overwrite_action
            selected_qs = torch.zeros(len(selected_actions))
        
        # conduct actions for batch
        #print(selected_qs)

        for iact, _ in enumerate(Index[Active]):
            Tidx = rTurn%3

            action = selected_actions[iact]
            #print(d, action)
            Q = selected_qs[iact]

            playerstate = sample_states[_][Tidx]

            if Forcemove[_]:
                Forcemove[_] = False

            # conduct a move
            myst = state2str(playerstate.sum(dim=0).numpy())
            cA = Counter(myst)
            cB = Counter(action[0])
            newst = ''.join(list((cA - cB).elements()))
            newunavail = unavail[_] + action[0]
            newhist = torch.roll(history[_],1,dims=0)
            newhist[0] = str2state_compressed(action[0])# str2state(action[0]).sum(axis=-2,keepdims=True) # first row is newest, others are moved downward

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

            sample_states[_][Tidx] = nextstate
            unavail[_] = newunavail
            history[_] = newhist
            lastmove[_] = newlast[_]

            results[_] = Q

            if len(newst) == 0:
                Active[_] = False
                results[_] = 0
                if Turn%3 == 0 and rTurn%3 == 0:
                    results[_] = 1
                if Turn%3 != 0 and rTurn%3 != 0:
                    results[_] = 1
                #print(action, rTurn, results[_])
        d += 1
        rTurn += 1

    #SLM.to('cpu')
    #QV.to('cpu')

    #print(results)
    return results

#@profile
def get_action_adv_batch(Turn, SLM, QV, Initstates, unavail, lastmove, Forcemove, history, temperature, Npass, Cpass, nAct=5, nRoll=10, ndepth=3, risk_penalty=0.0, maxtime=999999, sleep=True):
    
    selfplay_device = 'cuda'

    SLM.to(selfplay_device)
    QV.to(selfplay_device)

    t0 = time.time()

    player = Initstates[Turn%3]#.clone().detach()
    visible = Initstates[-1]#.clone().detach()

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((3,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.sum(axis=-2,keepdims=True).unsqueeze(0),
                            str2state(unavail).sum(axis=-2,keepdims=True).unsqueeze(0),
                            CC.unsqueeze(1),
                            visible.sum(axis=-2,keepdims=True).unsqueeze(0), # new feature
                            torch.zeros((1,15)).unsqueeze(0) + Turn%3, # role feature
                            history.sum(axis=-2,keepdims=True)])
    #print(Bigstate)
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput.to(selfplay_device)).to('cpu')
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    # get all actions

    acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)
    #print(state2str(player.sum(axis=0)))
    

    # generate inputs 2
    model_inter2 = torch.concat([hinput[:,0].sum(dim=-2),
                                hinput[:,7].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter2.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2.to(selfplay_device)).to('cpu').flatten()

    # get N best actions to sample from!
    N = min(nAct,len(acts))

    if temperature == 0:
        top_n_indices = torch.topk(output, N).indices
        n_actions = [acts[idx] for idx in top_n_indices]
        n_Q_values = output[top_n_indices]
    else:
        probabilities = torch.softmax(output / temperature, dim=0)
        distribution = torch.distributions.Categorical(probabilities)
        sampled_indices = distribution.sample((N,))
        n_actions = [acts[idx] for idx in sampled_indices]
        n_Q_values = output[sampled_indices]

    #print(n_actions)

    new_Q = [[] for i in range(N)]
    r_count = 0
    if N == 1:
        new_Q = n_Q_values
        if sleep:
            time.sleep(maxtime)
    else:
        # batch version
        #print('SSSSSS')
        dynamic_Roll = int(nAct / N * nRoll)

        nloop = int(dynamic_Roll / int(nAct / N * 10))

        for _ in range(nloop):

            if time.time() - t0 > maxtime:
                break

            # generate random states equal to X actions and r rolls

            rstates = []
            ractions = []
            for i in range(N) : # resample given action
                
                for r in range(int(nAct / N * 10)):

                    action = n_actions[i]
                    ractions.append(action.copy())

                    # construct fake initsates
                    sample1, sample2 = resample_state(Turn%3, Initstates, unavail, model_inter)
                    sample_states = [None,None,None,Initstates[-1].clone()]
                    sample_states[Turn%3] = Initstates[Turn%3].clone()
                    sample_states[(Turn-1)%3] = str2state(''.join([r2c_base[i]*sample1[i] for i in range(15)]))
                    sample_states[(Turn+1)%3] = str2state(''.join([r2c_base[i]*sample2[i] for i in range(15)]))
                    #print(sample1)
                    rstates.append(sample_states)
            #print(rstates[0][1],rstates[1][1])
            #quit()
            
            Qroll = rollout_batch(Turn, SLM, QV, rstates, ractions, unavail, lastmove.copy(), Forcemove, history.clone(), temperature, Npass, Cpass,
                                depth=ndepth)
            #print(Qroll)
            Qroll = Qroll.reshape(N, r+1)#.sum(dim=-1)
            for i in range(N):
                new_Q[i].append(Qroll[i])

            #new_Q += Qroll

            r_count += r+1
        for i in range(N):
            e = torch.concat(new_Q[i])
            #print(torch.mean(e),torch.var(e))
            new_Q[i] = torch.mean(e) - risk_penalty*torch.var(e)
        #new_Q /= r_count
        #print(new_Q)
        print(f'Scanned {N} actions, {r_count} bootstraps, depth {ndepth}.',end='\r')
        #quit()
    new_Q = torch.as_tensor(new_Q)
    best_action = n_actions[torch.argmax(new_Q)]
    Q = torch.max(new_Q)
    #print(n_Q_values.numpy().round(2), new_Q.numpy().round(2))
    #print(Turn, n_actions[0],best_action)
    SLM.to('cpu')
    QV.to('cpu')

    return best_action, Q



def game(Models, temperature, pause=0.5, nhistory=6, p_adv=[0], nAct=5, nRoll=10, ndepth=3, seed=0): # Player is 0, 1, 2 for L, D, U
    random.seed(seed)

    Init_states = init_game_3card() # [Lstate, Dstate, Ustate]

    Qs = []
    Q0 = []

    unavail = ''
    if Models[0].non_hist == 7:
        history = torch.zeros((nhistory,1,15))
    else:
        history = torch.zeros((nhistory,4,15))
    lastmove = ['',(0,0)]

    Turn = 0
    Npass = 0 # number of pass applying to rules
    Cpass = 0 # continuous pass

    Forcemove = True # whether pass is not allowed

    Log = f'Seed: {seed}\n'

    SLM,QV = Models

    while True: # game loop
        ts = time.time()
        # get player
        #print(Turn, lastmove)
        player = Init_states[Turn%3]
        
        if Turn%3 in p_adv:
            action, Q = get_action_adv_batch(Turn, SLM,QV,Init_states,unavail,lastmove, Forcemove, history, temperature, Npass, Cpass,
                                       nAct, nRoll, ndepth, maxtime=10, sleep=False)
        else:
            action, Q = get_action_serial_V2_2_2(Turn, SLM,QV,Init_states,unavail,lastmove, Forcemove, history, temperature, False)

        if Turn < 3:
            Q0.append(Q.item())

        #print(Turn, action, Q.item())

        if Forcemove:
            Forcemove = False

        # conduct a move
        myst = state2str(player.sum(dim=0).numpy())
        cA = Counter(myst)
        cB = Counter(action[0])
        newst = ''.join(list((cA - cB).elements()))
        newunavail = unavail + action[0]
        newhist = torch.roll(history,1,dims=0)
        if SLM.non_hist == 7:
            newhist[0] = str2state(action[0]).sum(axis=-2,keepdims=True) # first row is newest, others are moved downward
        else:
            newhist[0] = str2state(action[0])
        
        play = action[0]
        if action[1][0] == 0:
            play = 'pass'
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

        Log += f"{Label[Turn % 3]} {str(Turn).zfill(2)}    {myst.zfill(20).replace('0', ' ')} {play.zfill(20).replace('0', ' ')} by {Label[Turn % 3]}    {str(round(Q.item()*100,1)).zfill(5)}%\n"
        if Cpass == 2:
            Log += '\n'
        
        # update
        nextstate = str2state(newst)
        Init_states[Turn%3] = nextstate
        unavail = newunavail
        history = newhist
        lastmove = newlast
        
        time.sleep(max(pause - (time.time()-ts),0))

        if len(newst) == 0:
            break

        Turn += 1

    if Turn %3 == 0:
        Log += f'\nLandlord Wins'
    else:
        Log += f'\nFarmers Win'

    return Turn, Qs, Log


def play_game(seed,SLM,QV,plist):
    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    with torch.no_grad():
        Turn, Qs, Log = game([SLM, QV], 0, 0, 15, plist, 5, 100, 6, int(seed))
        print('E')
        return Turn % 3 == 0

if __name__ == '__main__':
    
    from multiprocessing import Pool

    

    wd = os.path.dirname(__file__)

    #if torch.get_num_threads() > 1: # no auto multi threading
    #    torch.set_num_threads(1)
    #    torch.set_num_interop_threads(1)

    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    name = 'H15-V2_2.2'

    mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if name + '_' in f]

    if len(mfiles) == 0:
        v_M = f'{name}_{str(0).zfill(10)}'
    else:
        v_M = f'{name}_{str(max(mfiles)).zfill(10)}'
    print('Model version:', v_M)

    N_history = int(v_M[1:3])
    SLM = Network_Pcard_V2_1(15+7, 7, y=1, x=15, lstmsize=512, hiddensize=1024)
    QV = Network_Qv_Universal_V1_1(6,15,1024)

    SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{v_M}.pt')))
    QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{v_M}.pt')))
    SLM.eval()
    QV.eval()

    print('- Model Loaded')

    

    LW = 0
    FW = 0
    N_game = 100
    np.random.seed(123456)
    seeds = np.random.randint(-1000000000, 1000000000, N_game)

    args_list = [(seed, SLM,QV,[0]) for seed in seeds]

    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    with torch.no_grad():
        Turn, Qs, Log = game([SLM, QV], 0, 0, 15, [0], 5, 100, 6, int(seeds[0]))

    quit()

    with Pool(12) as pool:
        results = pool.starmap(play_game, args_list)

    for result in results:
        if result:
            LW += 1
        else:
            FW += 1

    print(LW, FW)