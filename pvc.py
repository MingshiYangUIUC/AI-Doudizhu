import torch
import random
from collections import Counter

from base_funcs import *
from model import *

import os
import sys

import argparse

from datetime import datetime

def initialize_difficulty(iPlayer, Models, nhistory, difficulty):

    print(f'- Preparing game with difficulty Level {difficulty}...')
    i=0
    while True and i < 1000:
        i+=1
        Init_states = init_game() # [Lstate, Dstate, Ustate]
        Deck = [i.clone().detach() for i in Init_states]

        Qs = []
        Q0 = []
        Qout= 0

        unavail = ''
        history = torch.zeros((nhistory,4,15))
        lastmove = ['',(0,0)]

        Turn = 0
        Npass = 0 # number of pass applying to rules
        Cpass = 0 # continuous pass

        Forcemove = True # whether pass is not allowed

        while Turn < iPlayer+1: # game loop
            # get player
            #print(Turn, lastmove)
            player, model = Init_states[Turn%3], Models[Turn%3]

            # get card count
            card_count = [int(p.sum()) for p in Init_states]
            #print(card_count)
            CC = torch.zeros((4,15))
            CC[0][:min(card_count[0],15)] = 1
            CC[1][:min(card_count[1],15)] = 1
            CC[2][:min(card_count[2],15)] = 1
            #print(CC)

            # get action
            Bigstate = get_Bigstate(player,unavail,CC,history)
            action,Q = get_action_softmax(Bigstate,lastmove,model,Forcemove,0,'cpu',True)
            if Turn < 3:
                Q0.append(Q.item())
            Qs.append(Q.item())
                
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
            
            if action[1][0] == 0:
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
            Init_states[Turn%3] = nextstate
            unavail = newunavail
            history = newhist
            lastmove = newlast

            if len(newst) == 0:
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
        return init_game()
    print(f'- Done! ({i}) \n')
    return Deck

def gamewithplayer(iPlayer, Models, epsilon, pause=0.5, nhistory=6, automatic=False, difficulty=None, showall=False): # Player is 0, 1, 2 for L, D, U
    if difficulty is None:
        Init_states = init_game() # [Lstate, Dstate, Ustate]
    else:
        Init_states = initialize_difficulty(iPlayer, Models, nhistory, difficulty)
    
    Qs = []
    Q0 = []

    signs = ['-','+']

    unavail = ''
    history = torch.zeros((nhistory,4,15))
    lastmove = ['',(0,0)]

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
    Game += ','.join([state2str(st.sum(dim=0).numpy()) for st in Init_states])
    Game += '|'

    while True: # game loop
        ts = time.time()
        # get player
        #print(Turn, lastmove)
        player, model = Init_states[Turn%3], Models[Turn%3]

        # get card count
        card_count = [int(p.sum()) for p in Init_states]
        #print(card_count)
        CC = torch.zeros((4,15))
        CC[0][:min(card_count[0],15)] = 1
        CC[1][:min(card_count[1],15)] = 1
        CC[2][:min(card_count[2],15)] = 1
        #print(CC)

        # get action
        Bigstate = get_Bigstate(player,unavail,CC,history)
        action,Q = get_action_softmax(Bigstate,lastmove,model,Forcemove,epsilon,'cpu',True)
        action_suggest = action
        if Turn < 3:
            Q0.append(Q.item())
            Qd = 0.0
        else:
            Qd = Q.item() - Qs[Turn-3]
        Qs.append(Q.item())
        if Turn % 3 == iPlayer and not automatic:
            #print(f'Your cards: {state2str(player)}')
            all_acts = avail_actions(lastmove[0],lastmove[1],player,Forcemove)
            all_acts = [[''.join(sorted(a[0])),a[1]] for a in all_acts]
            while True:
                if len(all_acts) > 1 or Forcemove:
                    a = input(f"{Label[Turn%3]}: You have: {state2str(player.sum(dim=0).numpy())}. Your move: ")
                else:
                    a = input(f"{Label[Turn%3]}: You have: {state2str(player.sum(dim=0).numpy())}. No Larger Combo.")
                action = cards2action(a.upper().replace('1','A'))
                if action is not None:
                    action = [''.join(sorted(action[0])),action[1]]
                    if action in all_acts:
                        action = [''.join(sorted(action[0], key=lambda x: c2r_base[x])),action[1]]
                        ts = time.time()
                        break
                
                print('Illegal Action, try again!')
            
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

        Game += f'{Turn%3}:{play}|'

        Log += f"{Label[Turn % 3]} {str(Turn).zfill(2)}    {myst.zfill(20).replace('0', ' ')} {play.zfill(20).replace('0', ' ')} by {Label[Turn % 3]}    {str(round(Q.item()*100,1)).zfill(5)}%\n"
        if Cpass == 2:
            Log += '\n'
        if verbose:
            if Turn % 3 == iPlayer or automatic:
                if automatic or (showall and Turn%3==iPlayer):
                    outstr = f"{Label[Turn % 3]} {str(Turn).zfill(2)}     {myst.zfill(20).replace('0', ' ')} {play.zfill(20).replace('0', ' ')} by {Label[Turn % 3]} "
                    outstr += f'{str(round(Q.item()*100,1)).zfill(5)}% {signs[Qd>=0]}{str(round(abs(Qd)*100,1)).zfill(4)}%'
                    if showall:
                        outstr += f'  {action_suggest[0]}'
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
        print(''.join([Label[i]+'        '+state2str(p.sum(dim=0).numpy()).zfill(20).replace('0',' ')+'\n' for i,p in enumerate(Init_states) if i != Turn%3]))
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
    parser.add_argument("-s", "--showall", action="store_true", help="Show all statistics regardless of game mode.")
    parser.add_argument("-r", "--role", type=int, choices=[0, 1, 2], help="Role number. 0: Landlord, 1: Farmer-0, 2: Farmer-1. ")
    parser.add_argument("-t", "--temperature", type=float, help="Softmax temperature: randomness of AI actions (float >= 0)")
    parser.add_argument("-d", "--difficulty", type=int, choices=[1, 2, 3, 4, 5], help="Difficulty level as quality of initial cards. 1: excellent, 2: good, 3: fair, 4: poor, 5: terrible.")
    parser.add_argument("-p", "--pausetime", type=float, help="Pause after each move in seconds (float >= 0)")

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

    # Automatic mode check
    if args.automatic:
        print("Automatic mode is enabled.")
    else:
        print("Automatic mode is not enabled.")
    if args.showall:
        print("Show all stats.")
    
    return args.role, args.automatic, args.showall, args.temperature, args.difficulty, args.pausetime


if __name__ == '__main__':

    wd = os.path.dirname(__file__)

    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    v_M = 'H15-V4_0040000000'
    N_history = int(v_M[1:3])

    LM, DM, UM = Network_V2(N_history+4),Network_V2(N_history+4),Network_V2(N_history+4)
    
    LM.load_state_dict(torch.load(os.path.join(wd,'models',f'LM_{v_M}.pt')))
    DM.load_state_dict(torch.load(os.path.join(wd,'models',f'DM_{v_M}.pt')))
    UM.load_state_dict(torch.load(os.path.join(wd,'models',f'UM_{v_M}.pt')))

    LM.eval()
    DM.eval()
    UM.eval()

    player, automatic, showall, temperature, difficulty, pause = main()
    print('- Model Loaded')

    print(f'- Type in any combination of your available cards, press enter to play.\n  Press enter directly to pass.')
    print('- 10 is X, small joker is B, large joker is R.\n')
    if not automatic:
        print(f'- You are playing as {Label[player]}.\n')
    if automatic:
        print(f'- You are spectating AI playing, you set difficulty for {Label[player]}.\n')

    with torch.no_grad():
        Turn, Qs, Log = gamewithplayer(player,[LM,DM,UM],temperature,pause,N_history, automatic, difficulty, showall)

    #print(Log)

'''
e:
conda activate rl-0
python E:\\Documents\\ddz\\pvc.py -h
'''
