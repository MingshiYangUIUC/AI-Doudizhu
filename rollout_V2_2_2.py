from model_V2 import *
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
            probabilities = sample2 / sample2.sum()
            chosen_card = np.random.choice(15, p=probabilities)
            sample1[chosen_card] += 1
            sample2[chosen_card] -= 1
    #print(sample2)
    return sample1, sample2

def rollout_2_2_2(Turn, SLM, QV, sample_states, overwrite_action, unavail, lastmove, Forcemove, history, temperature, Npass, Cpass, depth=3):

    rTurn = Turn
    newlast = lastmove

    d = 0
    end = False

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
            model_inter = SLM(hinput)
            role = torch.zeros((model_inter.shape[0],15)) + rTurn%3
            acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)
            model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                        hinput[:,7].sum(dim=-2),
                                        model_inter,
                                        role],dim=-1)
            model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])
            # get q values
            output = QV(model_input2).flatten()

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
    
    return Q


def get_action_adv(Turn, SLM, QV, Initstates, unavail, lastmove, Forcemove, history, temperature, Npass, Cpass, nAct=5, nRoll=10, ndepth=3, maxtime=999999):
    
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
    if N == 1:
        new_Q = n_Q_values
    else:
        for i in range(N): # resample given action
            action = n_actions[i]
            Qroll = [n_Q_values[i]] # original value has some weight (good if simulation number is small)

            while time.time() - t0 < maxtime and len(Qroll)-1 < nRoll:
            #for r in range(nRoll): # construct fake initsates, and rollout
                sample1, sample2 = resample_state(Turn%3, Initstates, unavail, model_inter)
                sample_states = [None,None,None,Initstates[-1].clone()]
                sample_states[Turn%3] = Initstates[Turn%3].clone()
                sample_states[(Turn-1)%3] = str2state(''.join([r2c_base[i]*sample1[i] for i in range(15)]))
                sample_states[(Turn+1)%3] = str2state(''.join([r2c_base[i]*sample2[i] for i in range(15)]))
                Qroll.append(
                    rollout_2_2_2(Turn, SLM, QV, sample_states, action.copy(), unavail, lastmove.copy(), Forcemove, history.clone(), temperature, Npass, Cpass,
                                depth=ndepth))
                
            Qroll = np.array(Qroll)
            #print(Turn, n_Q_values[i].item(), np.mean(Qroll))
            #print(np.round(Qroll,1))
            new_Q[i] = np.mean(Qroll)

    best_action = n_actions[torch.argmax(new_Q)]
    Q = torch.max(new_Q)
    #print(n_Q_values.numpy().round(2), new_Q.numpy().round(2))
    #print(Turn, n_actions[0],best_action)
    return best_action, Q