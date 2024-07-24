from model_utils import *
from base_utils import *

def resample_state(Turn, Initstates, unavail, model_inter):
    cprob = model_inter.detach().numpy().reshape(2,15)

    cprob[0] = cprob[0] / np.sum(cprob[0])
    cprob[1] = cprob[1] / np.sum(cprob[1])
    
    # total count is exact values
    total_count = Initstates[Turn].detach().numpy().flatten()
    total_count += unavail.numpy().flatten()

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
            prob = np.ones(15)
            prob[sample2==0] = 0
            prob /= prob.sum()
            chosen_card = np.random.choice(15, p=prob)
            sample1[chosen_card] += 1
            sample2[chosen_card] -= 1

    return sample1, sample2


def rollout_serial(Turn, SLM, QV, sample_states, overwrite_action, unavail, played_cards, lastmove, Forcemove, history, temperature, Npass, Cpass, depth=3):

    rTurn = Turn
    newlast = lastmove

    d = 0
    end = False

    while True and d <= depth:

        player = sample_states[rTurn%3]
        #print(rTurn%3,[ss.shape for ss in sample_states],player.shape)

        if d > 0: # rollout

            #player = Initstates[Turn%3]#.clone().detach()
            visible = sample_states[-1]#.clone().detach()

            # get action
            Bigstate = torch.cat([player.unsqueeze(0),
                                    unavail.unsqueeze(0),
                                    played_cards,
                                    visible.unsqueeze(0), # new feature
                                    torch.full((1, 15), Turn%3),
                                    history])
            Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
            # generate inputs
            hinput = Bigstate.unsqueeze(0)

            model_inter, lstm_out = SLM(hinput)

            # get all actions
            acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)

            # generate inputs 2
            model_inter = torch.concat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                        model_inter, # upper and lower states
                                        #role,
                                        lstm_out, # lstm encoded history
                                        ],dim=-1)
            model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

            # get q values
            output = QV(model_input2).flatten()

            Q = torch.max(output).item()
            best_act = acts[torch.argmax(output)]

            action = best_act

        else: # use action
            action = overwrite_action
            Q = 0.5

        if Forcemove:
            Forcemove = False

        # conduct a move
        newhiststate = str2state_1D(action[0])
        newst = player - newhiststate
        newunavail = unavail + newhiststate
        newhist = torch.roll(history,1,dims=0)
        newhist[0] = newhiststate
        
        played_cards[rTurn%3] += newhiststate

        play = action[0]
        if action[1][0] == 0:
            play = 'pass'
            Cpass += 1
            if Npass < 1:
                Npass += 1
            else:
                #print('Clear Action')
                newlast = ('',(0,0))
                Npass = 0
                Forcemove = True
        else:
            newlast = action
            Npass = 0
            Cpass = 0
        
        # update
        nextstate = newst
        sample_states[rTurn%3] = nextstate
        unavail = newunavail
        #print(newlast)
        history = newhist
        lastmove = newlast
        
        #print(Q)

        if newst.max() == 0:
            end = True
            break

        rTurn += 1
        d += 1
    #print(d, rTurn)
    #W = 0
    if end:
        Q = 0.0
        if rTurn%3 == 0 and Turn%3 == 0:
            Q = 1.0
        if rTurn%3 != 0 and Turn%3 != 0:
            Q = 1.0


    return Q


def get_action_adv(Turn, SLM, QV, Initstates, unavail, played_cards, lastmove, Forcemove, history, temperature, Npass, Cpass, nAct=5, nRoll=10, ndepth=3, maxtime=6, sleep=False):
    
    t0 = time.time()

    player = Initstates[Turn%3]#.clone().detach()
    visible = Initstates[-1]#.clone().detach()

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            unavail.unsqueeze(0),
                            played_cards,
                            visible.unsqueeze(0), # new feature
                            torch.full((1, 15), Turn%3),
                            history])
    Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
    # generate inputs
    hinput = Bigstate.unsqueeze(0)

    model_interX, lstm_out = SLM(hinput)

    # get all actions
    acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)

    # generate inputs 2
    model_inter = torch.concat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                model_interX, # upper and lower states
                                #role,
                                lstm_out, # lstm encoded history
                                ],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()

    # get N best actions to sample from!
    N = min(nAct,len(acts))


    top_n_indices = torch.topk(output, N).indices
    n_actions = [acts[idx] for idx in top_n_indices]
    n_Q_values = output[top_n_indices]


    new_Q = torch.zeros(N)
    r_count = 0
    if N == -1:
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
                sample1, sample2 = resample_state(Turn%3, Initstates, unavail, model_interX)
                sample_states = [None,None,None,Initstates[-1].clone()]
                sample_states[Turn%3] = Initstates[Turn%3].clone()
                sample_states[(Turn-1)%3] = str2state_1D(''.join([r2c_base[i]*sample1[i] for i in range(15)]))
                sample_states[(Turn+1)%3] = str2state_1D(''.join([r2c_base[i]*sample2[i] for i in range(15)]))
                Qroll.append(
                    rollout_serial(Turn, SLM, QV, sample_states, action.copy(), unavail.clone(), played_cards.clone(), lastmove, Forcemove, history.clone(), temperature, Npass, Cpass,
                                depth=ndepth))

            Qroll = torch.as_tensor(Qroll)

            new_Q += Qroll

            r_count += 1
        new_Q /= r_count
    
        print(f'Scanned {N} actions, {r_count} bootstraps, depth {ndepth}.')
    print('Bootstrap:  '+' | '.join([f'{"pass" if n_actions[i][0] == "" else n_actions[i][0]} {str(round(float(new_Q[i])*100,1)).zfill(5)}%' for i in range(N)]))
    best_action = n_actions[torch.argmax(new_Q)]
    Q = torch.max(new_Q)
    #print(n_Q_values.numpy().round(2), new_Q.numpy().round(2))
    #print(Turn, n_actions[0],best_action)
    return best_action, Q