'''

STILL UNDER CONSTRUCTION

This module helps player using trained model...

Player enters current cards holding by self, landlord's visible cards, and sequential historical actions in string format.
AI returns next action to be taken by the player using available information.

'''

import torch
import random
from collections import Counter
from model_utils import *
from base_utils import *
from rollout import *
import os
import sys
import argparse
from datetime import datetime



if __name__ == '__main__':

    wd = os.path.dirname(__file__)

    if torch.get_num_threads() > 1: # no auto multi threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    Label = ['Landlord','Farmer-0','Farmer-1'] # players 0, 1, and 2

    name = 'H15-V2_2.3'
    name = 'H15-V2_3.0'
    mfiles = [int(f.split('_')[-1][:-3]) for f in os.listdir(os.path.join(wd,'models')) if '_'+name+'_' in f]

    v_M = f'{name}_{str(max(mfiles)).zfill(10)}'
    print('Model version:', v_M)

    #SLM = Network_Pcard_V2_1_BN(15+7, 7, y=1, x=15, lstmsize=512, hiddensize=512)
    #QV = Network_Qv_Universal_V1_1_BN(6,15,512)
    SLM = Network_Pcard_V2_2_BN_dropout(15+7, 7, y=1, x=15, lstmsize=256, hiddensize=512)
    QV = Network_Qv_Universal_V1_2_BN_dropout(11*15,256,512)
    SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{v_M}.pt')))
    QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{v_M}.pt')))
    SLM.eval()
    QV.eval()

    parser = argparse.ArgumentParser(description="Process the command-line options")

    # Add arguments
    parser.add_argument("-c", "--custom", action="store_true", help="Enable custom input, you can overwrite AI suggestion.")
    parser.add_argument("-sh", "--showall", action="store_true", help="Show SL prediction by model.")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Softmax temperature: randomness of AI actions (float >= 0)")
    parser.add_argument("-th", "--thinktime", type=float, default=0.0, help="AI thinktime (float >= 0)")
    parser.add_argument("-rp", "--riskpenalty", type=float, default=0.0, help="AI risk penalty (float >= 0)")
    parser.add_argument("-r", "--role", type=int, choices=[0, 1, 2], help="Role number. 0: Landlord, 1: Farmer-0, 2: Farmer-1. ")

    # Parse arguments
    args = parser.parse_args()

    thinktime = args.thinktime
    risk_penalty = args.riskpenalty
    temperature = args.temperature
    custom = args.custom
    show_sl = args.showall
    if args.role is None:
        role = int(input('Your role (0:L, 1:F0, 2:F1): '))
    else:
        role = args.role
    # Construct required model params prior to players turn
    gamestate = init_game_3card()

    unavail = torch.zeros(15)
    played_cards = torch.zeros((3,15))
    history = torch.zeros((15,15))
    lastmove = ('',(0,0))

    

    gamestate[role] = str2state_1D(input('Your cards: ').upper().replace('1','A'))
    # initialize opponent as card count at first entry
    for i in range(3):
        if i != role:
            gamestate[i] = gamestate[i] * 0 + torch.sum(gamestate[i])
            gamestate[i][1:] = 0
    gamestate[-1] = str2state_1D(input('Public cards: ').upper().replace('1','A'))

    
    #print(gamestate)
    
    Turn = 0
    Npass = 0 # number of pass applying to rules
    Cpass = 0 # continuous pass
    Forcemove = True # whether pass is not allowed

    while True:#Turn < playerrole + 100:

        # record opponent move
        if Turn % 3 != role:
            histmove = input(f'The move at turn {Turn} {Label[Turn%3]}: ').upper().replace('1','A')
            action = cards2action(histmove.upper().replace('1','A'))
            # remove card count
            gamestate[Turn % 3][0] -= len(action[0])
        
        # suggest move!
        else:
            with torch.inference_mode():
                if 'V2_2' in name:# back compatible
                    if thinktime == 0:
                        action, Q = get_action_serial_V2_2_2(Turn, SLM,QV,gamestate,unavail,lastmove, Forcemove, history, temperature, show_sl)
                    else:
                        action, Q = get_action_adv_batch_mp(Turn, SLM, QV, gamestate,unavail,lastmove, Forcemove, history, temperature, Npass, Cpass,
                                        nAct=6, nRoll=400, ndepth=36, risk_penalty=risk_penalty, maxtime=thinktime, nprocess=12, sleep=True)
                elif 'V2_3' in name:
                    action, Q = get_action_serial_V2_3_0(Turn, SLM,QV,gamestate,unavail,played_cards,lastmove, Forcemove, history, temperature, show_sl)
            # else:
            if action[0] == '':
                sp = 'pass'
            else:
                sp = action[0]
            print('Suggested:',sp)
            print('WR:',f'{str(round(Q.item()*100,1)).zfill(5)}%')
            
            if custom:
                histmove = input(f'Your custom move? (type your move or "enter" to pass): ').upper().replace('1','A')
                if len(histmove) > 0:
                    action = cards2action(histmove.upper().replace('1','A'))

        if Forcemove:
            Forcemove = False

        newhiststate = str2state_1D(action[0])

        if Turn % 3 == role:
            gamestate[role] = gamestate[role] - newhiststate
        
        newunavail = unavail + newhiststate
        newhist = torch.roll(history,1,dims=0)
        newhist[0] = newhiststate# first row is newest, others are moved downward
        played_cards[Turn % 3] += newhiststate
        
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
        
        unavail = newunavail
        history = newhist
        lastmove = newlast

        if gamestate[Turn%3].max() == 0:#(Turn % 3 == playerrole and ):

            break
        #elif len(input('Game end? (type any if game end) ')) > 0:
        #    break

        Turn += 1

        #print(gamestate)