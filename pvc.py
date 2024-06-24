import torch
import random
from collections import Counter

#from base_funcs_V2 import *
from model_utils import *

from base_utils import *

from rollout import *

import os
import sys

import argparse

from datetime import datetime

def initialize_difficulty(iPlayer, Models, nhistory, difficulty, bomb=False):

    print(f'- Preparing game with difficulty Level {difficulty}...')
    i=0

    SLM,QV = Models

    while True and i < 1000:
        i+=1
        if not bomb:
            Init_states = init_game_3card() # [Lstate, Dstate, Ustate]
        else:
            Init_states = init_game_3card_bombmode(bstrength)
        public_cards = state2str_1D(Init_states[-1].numpy())

        Deck = [i.clone().detach() for i in Init_states]

        Qs = []
        Q0 = []
        Qout= 0

        unavail = torch.zeros(15)
        history = torch.zeros((nhistory,15))
        lastmove = ('',(0,0))

        Turn = 0
        Npass = 0 # number of pass applying to rules
        Cpass = 0 # continuous pass

        Forcemove = True # whether pass is not allowed

        while Turn < iPlayer+1: # game loop
            # get player
            #print(Turn, lastmove)
            player = Init_states[Turn%3]


            #print(CC)

            action, Q = get_action_serial_V2_2_2(Turn, SLM,QV,Init_states,unavail,lastmove, Forcemove, history, temperature)
            

            if Turn < 3:
                Q0.append(Q.item())
            Qs.append(Q.item())
                
            if Forcemove:
                Forcemove = False

            # conduct a move
            # myst = state2str_1D(player)
            newhiststate = str2state_1D(action[0])
            newst = player - newhiststate
            newunavail = unavail + newhiststate
            newhist = torch.roll(history,1,dims=0)
                
            if action[1][0] == 0:
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
            Init_states[Turn%3] = nextstate
            unavail = newunavail
            history = newhist
            lastmove = newlast

            if newst.max() == 0:
                break

            Turn += 1
        #Qout = 1/3*(Q0[0]+2-Q0[1]-Q0[2])
        Qout = Q0[-1]
        #if iPlayer > 0:
        #    Qout = 1-Qout
        #print(f'Your initial win rate is {round(Q0*100,1)}%.')
        if (1-Qout)*100 // 20 + 1 == difficulty:
            break
    if i >= 1000:
        print('Sadly it is hard to setup that difficulty, try random game!')
        return init_game_3card()
    print(f'- Done! ({i}) \n')
    return Deck

def gamewithplayer(iPlayer, Models, temperature, pause=0.5, nhistory=6, automatic=False, difficulty=None,\
                   showall=False, bomb=False, thinktime=0, thinkplayer='012', risk_penalty=0.0, seed=None): # Player is 0, 1, 2 for L, D, U
    if seed is None:
        seed = np.random.randint(-100000000,100000000)
    else:
        if difficulty is not None:
            print('Seed will not work correctly together with difficulty.')
        
    print('Game Seed:', seed)
    random.seed(seed)
    
    if difficulty is None:
        if not bomb:
            Init_states = init_game_3card() # [Lstate, Dstate, Ustate]
        else:
            Init_states = init_game_3card_bombmode(bstrength)
    else:
        if not bomb:
            Init_states = initialize_difficulty(iPlayer, Models, nhistory, difficulty, bomb)
        else:
            print('It is hard to guess initial difficulty in Bomb Mode!')
            Init_states = init_game_3card_bombmode(bstrength)

    public_cards = state2str_1D(Init_states[-1].numpy())
    print(f'Landlord Cards: {public_cards}')

    Qs = []
    Q0 = []

    signs = ['-','+']

    unavail = torch.zeros(15)
    history = torch.zeros((nhistory,15))

    lastmove = ('',(0,0))

    Turn = 0
    Npass = 0 # number of pass applying to rules
    Cpass = 0 # continuous pass
    verbose = 1

    Forcemove = True # whether pass is not allowed

    Log = ''

    Game = ''
    if not automatic:
        Game += f'Player:{iPlayer}|'
    else:
        Game += f'Player:A|'
    Game += ','.join([state2str_1D(st) for st in Init_states])
    Game += '|'

    SLM,QV = Models

    while True: # game loop
        ts = time.time()
        # get player
        #print(Turn, lastmove)
        player = Init_states[Turn%3]
        
        #print(CC)

        hint = showall and Turn%3==iPlayer
        
        if (Turn%3==iPlayer and not automatic) or thinktime == 0 or (str(Turn%3) not in thinkplayer):
            action, Q = get_action_serial_V2_2_2(Turn, SLM,QV,Init_states,unavail,lastmove, Forcemove, history, temperature,hint)
        else:
            action, Q = get_action_adv_batch_mp(Turn, SLM,QV,Init_states,unavail,lastmove, Forcemove, history, temperature, Npass, Cpass,
                                       nAct=8, nRoll=40, ndepth=36, risk_penalty=risk_penalty, maxtime=thinktime, nprocess=6, sleep=True)
            if thinktime > 0:
                #pause += thinktime
                ts = time.time()

        action_suggest = [a for a in action]
        #print(Q, action)
        if action_suggest[0] == '':
            action_suggest[0] = 'pass'

        if Turn % 3 == iPlayer and not automatic:
            #print(f'Your cards: {state2str(player)}')
            all_acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)
            all_acts = [[''.join(sorted(a[0])),a[1]] for a in all_acts]
            while True:
                if len(all_acts) > 1 or Forcemove:
                    a = input(f"{Label[Turn%3]}: You have: {state2str_1D(player.numpy())}. Your move: ")
                else:
                    a = input(f"{Label[Turn%3]}: You have: {state2str_1D(player.numpy())}. No Larger Combo.")
                action = cards2action(a.upper().replace('1','A'))
                if action is not None:
                    action = [''.join(sorted(action[0])),action[1]]
                    if action in all_acts:
                        action = [''.join(sorted(action[0], key=lambda x: c2r_base[x])),action[1]]
                        Q_r = evaluate_action_serial_V2_2_2(Turn, SLM,QV,Init_states,unavail,lastmove, Forcemove, history, temperature,
                                                                action[0])
                        ts = time.time()
                        break
                
                print('Illegal Action, try again!')
        else:
            Q_r = Q
        
        if Turn < 3:
            Q0.append(Q.item())
            Qd = 0.0
        else:
            Qd = Q_r.item() - Qs[Turn-3]
        Qs.append(Q_r.item())

        if Forcemove:
            Forcemove = False

        # conduct a move
        '''myst = state2str(player.sum(dim=0).numpy())
        cA = Counter(myst)
        cB = Counter(action[0])
        newst = ''.join(list((cA - cB).elements()))
        newunavail = unavail + action[0]
        newhist = torch.roll(history,1,dims=0)
        if SLM.non_hist == 7:
            newhist[0] = str2state(action[0]).sum(axis=-2,keepdims=True) # first row is newest, others are moved downward
        else:
            newhist[0] = str2state(action[0])'''
        myst = state2str_1D(player)
        newhiststate = str2state_1D(action[0])
        newst = player - newhiststate
        newunavail = unavail + newhiststate
        newhist = torch.roll(history,1,dims=0)
        #newhiststate = str2state_1D(action[0])# str2state(action[0]).sum(axis=-2,keepdims=True) 
        
        newhist[0] = newhiststate# first row is newest, others are moved downward
        
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

        Game += f'{Turn%3}:{play}|'

        Log += f"{Label[Turn % 3]} {str(Turn).zfill(2)}    {myst.zfill(20).replace('0', ' ')} {play.zfill(20).replace('0', ' ')} by {Label[Turn % 3]}    {str(round(Q.item()*100,1)).zfill(5)}%\n"
        if Cpass == 2:
            Log += '\n'
        if verbose:
            if Turn % 3 == iPlayer or automatic:
                if automatic or (showall and Turn%3==iPlayer):
                    outstr = f"{Label[Turn % 3]} {str(Turn).zfill(2)}     {myst.zfill(20).replace('0', ' ')} {play.zfill(20).replace('0', ' ')} by {Label[Turn % 3]} "
                    outstr += f'{str(round(Q_r.item()*100,1)).zfill(5)}% {signs[Qd>=0]}{str(round(abs(Qd)*100,1)).zfill(4)}%'
                    if showall:
                        if sorted(action_suggest[0]) != sorted(play):
                            outstr += f' | {str(round(Q.item()*100,1)).zfill(5)}% {action_suggest[0]}'
                        else:
                            outstr += f' | Best'
                    print(outstr)
                    #print(Label[Turn%3], str(Turn).zfill(2), '   ', myst.zfill(20).replace('0',' '), play.zfill(20).replace('0',' '), 'by', Label[Turn%3],
                    #       f'{str(round(Q.item()*100,1)).zfill(5)}% {signs[Qd>=0]}{str(round(abs(Qd)*100,1)).zfill(4)}%')
                else:
                    print(Label[Turn%3], str(Turn).zfill(2), '   ', myst.zfill(20).replace('0',' '), play.zfill(20).replace('0',' '), 'by', Label[Turn%3],)
            else:
                print(Label[Turn%3], str(Turn).zfill(2), '   ', ' '*(20-len(myst))+'_'*len(action[0])+'?'*(len(myst)-len(action[0])), play.zfill(20).replace('0',' '), 'by', Label[Turn%3],)
            if Cpass == 2:
                print('')


        # update
        nextstate = newst

        Init_states[Turn%3] = nextstate
        unavail = newunavail
        history = newhist
        lastmove = newlast
        
        time.sleep(max(pause - (time.time()-ts),0))

        if newst.max() == 0:
            break

        Turn += 1

    if Turn %3 == 0:
        Log += f'\nLandlord Wins'
    else:
        Log += f'\nFarmers Win'
    Game += f'W{Turn%3}'
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    f = open(os.path.join(wd,'games',f'G_{dt_string}.txt'),'w')
    f.write(Game)
    f.close()

    if verbose:
        if Turn %3 == 0:
            print(f'\nLandlord Wins')
        else:
            print(f'\nFarmers Win')
        print('\nCards Remaining:')
        print(''.join([Label[i]+'        '+state2str_1D(p).zfill(20).replace('0',' ')+'\n' for i,p in enumerate(Init_states[:-1]) if i != Turn%3]))
        if (not automatic) or showall:
            #Q0 = 1/3*(Q0[0]+2-Q0[1]-Q0[2])
            Q0 = Q0[iPlayer]
            #if iPlayer > 0:
            #    Q0 = 1-Q0
            print(f'Your initial win rate is roughly {round(Q0*100,1)}%.')
            if Q0 < 1/3:
                if (iPlayer>0) == (Turn%3>0):
                    print('You win, I thought it was unlikely for you to win. Good job.')
                else:
                    print('You lose, but it was fine since I got better cards.')
            elif Q0 > 2/3:
                if (iPlayer>0) == (Turn%3>0):
                    print('You win, but just because I got bad cards.')
                else:
                    print('You lose, I beat you even though you had better cards.')
            else:
                if (iPlayer>0) == (Turn%3>0):
                    print('You win, GGWP.')
                else:
                    print('You lose, GGWP.')

    return Turn, Qs, Log

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the command-line options")

    # Add arguments
    parser.add_argument("-a", "--automatic", action="store_true", help="Enable automatic mode: three AI playing and you just spectate one of them.")
    parser.add_argument("-b", "--bombmode", action="store_true", help="Enable bomb mode: players are more likely to get bombs.")
    parser.add_argument("-sh", "--showall", action="store_true", help="Show all statistics regardless of game mode.")
    parser.add_argument("-r", "--role", type=int, choices=[0, 1, 2], help="Role number. 0: Landlord, 1: Farmer-0, 2: Farmer-1. ")
    parser.add_argument("-t", "--temperature", type=float, help="Softmax temperature: randomness of AI actions (float >= 0)")
    parser.add_argument("-d", "--difficulty", type=int, choices=[1, 2, 3, 4, 5], help="Difficulty level as quality of initial cards. 1: excellent, 2: good, 3: fair, 4: poor, 5: terrible.")
    parser.add_argument("-p", "--pausetime", type=float, help="Pause after each move in seconds (float >= 0)")
    parser.add_argument("-th", "--thinktime", type=float, help="AI thinktime (float >= 0)")
    parser.add_argument("-tp", "--thinkplayer", type=str, help="The player who thinks (string containing 012)")
    parser.add_argument("-rp", "--riskpenalty", type=float, help="AI risk penalty (float >= 0)")
    parser.add_argument("-s", "--seed", type=int, help="Game Seed (int)")

    # Parse arguments
    args = parser.parse_args()

    # If no player number is provided, select a random one from 0, 1, 2
    if args.role is None:
        args.role = random.choice([0, 1, 2])
        print(f"No role number provided, randomly selected: {args.role}")
    else:
        print(f"role Number: {args.role}")
    if args.temperature is None:
        args.temperature = 0.0
    print(f'Softmax Temperature for AI: {args.temperature}')

    if args.pausetime is None:
        args.pausetime = 0.5
    if args.thinktime is None:
        args.thinktime = 0.0
    else:
        args.pausetime = 0.1
    if args.riskpenalty is None:
        args.riskpenalty = 0.0
    if args.thinkplayer is None: # set to everyone think
        args.thinkplayer = '012'

    # Automatic mode check
    if args.automatic:
        print("Automatic mode is enabled.")
    else:
        print("Automatic mode is not enabled.")
    if args.showall:
        print("Show all stats.")
    if args.bombmode:
        print("!!!!!!BOMB mode ENABLED!!!!!!")
    
    return args.role, args.automatic, args.showall, args.temperature, args.difficulty, args.pausetime, \
           args.thinktime, args.thinkplayer, args.riskpenalty, args.bombmode, args.seed


if __name__ == '__main__':

    wd = os.path.dirname(__file__)

    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    player, automatic, showall, temperature, difficulty, pause, thinktime, thinkplayer, rp, bomb, seed = main()

    name = 'H15-V2_2.2'
    name = 'H15-VBx5_128-128-128_0.01_0.0001-0.0001_256'
    if bomb:
        name += '-bomber'
        mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if name + '_' in f]
        if len(mfiles) == 0:
            name = name[:-7]
        #pass
    
    mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if name + '_' in f]
    
    if len(mfiles) == 0:
        v_M = f'{name}_{str(0).zfill(10)}'
    else:
        v_M = f'{name}_{str(max(mfiles)).zfill(10)}'
    #v_M = 'H15-V2_1.2-Contester_0022600000'
    print('Model version:', v_M)
    
    N_history = int(v_M[1:3])
    N_feature = 5
    #SLM = Network_Pcard_V1_1(20,5)
    #QV = Network_Qv_Universal_V1_1(6,15,512)
    #SLM = Network_Pcard_V2_1(15+7, 7, y=1, x=15, lstmsize=512, hiddensize=1024)
    #QV = Network_Qv_Universal_V1_1(6,15,1024)

    SLM = Network_Pcard_V2_1_BN(15+7, 7, y=1, x=15, lstmsize=128, hiddensize=128)
    QV = Network_Qv_Universal_V1_1_BN(6,15,128)
    
    SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{v_M}.pt')))
    QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{v_M}.pt')))
    SLM.eval()
    QV.eval()

    
    print('- Model Loaded')

    print(f'- Type in any combination of your available cards, press enter to play.\n  Press enter directly to pass.')
    print('- 10 is X, small joker is B, large joker is R.\n')
    if not automatic:
        print(f'- You are playing as {Label[player]}.\n')
    if automatic:
        print(f'- You are spectating AI playing, you set difficulty for {Label[player]}.\n')


    #random.seed(10000)
    with torch.no_grad():
        bstrength=200
        Turn, Qs, Log = gamewithplayer(player, [SLM,QV], temperature, pause, N_history, automatic,\
                                       difficulty, showall, bomb, thinktime, thinkplayer, rp, seed)

    #print(Log)

'''
e:
conda activate rl-0
python E:\\Documents\\ddz\\pvc.py -h
'''
