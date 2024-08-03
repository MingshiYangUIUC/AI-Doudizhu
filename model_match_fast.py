"""

Evaluate model performance by testing a series of models against a fixed gating model one at a time.

The model parameters should be defined in 'MATCH' section of 'config.ini'.
The model classes are defined in 'model_utils.py'.
Each model in the series plays with the gating model using the same randomly initialized games. Two models swap roles (farmer/landlord) and essentially play the same initialization twice.
Win rate of model series against gating model will be logged and played.
Testing can be parallelized if computational resource permits.

"""

import torch
import random
import model_utils
from base_funcs_selfplay import gating_batchpool
import os
import sys
import torch.multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
import gc
import argparse
import configparser
import statsmodels.api as sm

wd = os.path.join(os.path.dirname(__file__))

def calculate_win_rates_and_cis(wins_array, total_games_array, confidence_level=0.95):
    if isinstance(wins_array, int) and isinstance(total_games_array, int):
        wins_array = [wins_array]
        total_games_array = [total_games_array]
    win_rates = []
    margins_of_error = []
    
    for wins, total_games in zip(wins_array, total_games_array):
        # Calculate win rate
        win_rate = wins / total_games
        
        # Calculate the confidence interval
        conf_int = sm.stats.proportion_confint(wins, total_games, alpha=1-confidence_level, method='normal')
        
        # Calculate the margin of error
        margin_of_error = (conf_int[1] - conf_int[0]) / 2
        
        # Convert to percentages
        win_rate_percent = win_rate * 100
        margin_of_error_percent = margin_of_error * 100
        
        win_rates.append(win_rate_percent)
        margins_of_error.append(margin_of_error_percent)
    
    return win_rates, margins_of_error

def simulate_match_wrapper(m1, m2, t, device, nh, ng, ne, seed): # dual match
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    #m1, m2, t, device, nh, ng, ne, seed = args
    models = [m1, m2]
    torch.cuda.empty_cache()
    with torch.inference_mode():
        gatingresult = gating_batchpool(models, t, device, Nhistory=nh, ngame=ng, ntask=ne, rseed=seed)
    LW = (gatingresult==0).sum()
    torch.cuda.empty_cache()
    #gatingresult = gating_batchpool(models[::-1], t, device, Nhistory=nh, ngame=ng, ntask=ne, rseed=seed)
    #FW = (np.array(gatingresult)!=0).sum()
    gc.collect()
    return LW #, FW

def parse_args():
    parser = argparse.ArgumentParser(description="Run the training script with specified parameters.")
    
    # master arguments
    parser.add_argument('--v_gate', type=str, default='', help="Version of gating model")
    parser.add_argument('--mg_par0', type=int, default=512, help="Model parameter 0: SLM LSTM")
    parser.add_argument('--mg_par1', type=int, default=512, help="Model parameter 1: SLM MLP")
    parser.add_argument('--mg_par2', type=int, default=512, help="Model parameter 2: QV MLP")
    parser.add_argument('--i_gate', type=int, default='-1', help="Iteration number of gating model")
    parser.add_argument('--v_series', type=str, default='', help="Version of model series")
    parser.add_argument('--ms_par0', type=int, default=512, help="Model parameter 0: SLM LSTM")
    parser.add_argument('--ms_par1', type=int, default=512, help="Model parameter 1: SLM MLP")
    parser.add_argument('--ms_par2', type=int, default=512, help="Model parameter 2: QV MLP")
    parser.add_argument('--i_start', type=int, default='-1', help="Start iteration number")
    parser.add_argument('--i_stop', type=int, default='-1', help="Stop iteration number (inclusive)")
    parser.add_argument('--i_step', type=int, default='-1', help="Iteration number step")
    parser.add_argument('--n_game', type=int, default='-1', help="Number of games per dual")
    parser.add_argument('--n_processes', type=int, default='-1', help="Max number of CPU processes used in selfplay, could be lower if there are not enough tasks")
    parser.add_argument('--selfplay_batch_size', type=int, default='-1', help="Batch number of concurrent games send to GPU by each process")
    parser.add_argument('--selfplay_device', type=str, default='cpu', help="Device for selfplay games")

    # Add an argument for config file
    parser.add_argument('--config', type=str, default='.config.ini', help="Path to configuration file (relative)")

    args = parser.parse_args()
    #print(args)

    # If config file is provided, parse it and update arguments
    if args.config:
        config = configparser.ConfigParser()
        wd = os.path.dirname(__file__)
        config.read(os.path.join(wd,args.config))
        # Update arguments from config file
        if 'MATCH' in config:
            for key in vars(args):
                if key in config['MATCH']:
                    #print(key, config['MATCH'][key])
                    setattr(args, key, type(getattr(args, key))(config['MATCH'][key]))

    return args

if __name__ == '__main__':

    device = 'cpu'

    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    mp.set_start_method('spawn', force=True)
    
    # determine which models participate in the contest
    
    args = parse_args()

    players = [str(i).zfill(10) for i in range(args.i_start,args.i_stop+1,args.i_step)]
    num_processes = min(args.n_processes,(len(players))*2)
    nsim_perplayer = args.n_game
    nsim = nsim_perplayer
    players.append(str(args.i_gate).zfill(10))
    gate_player_index = len(players)-1 # other players play with this player, which is the last player
    versions = [args.v_series]
    eval_device = args.selfplay_device

    models = []
    nplayer = len(players)*len(versions)
    fullplayers = []

    for v in versions:
        fullplayers += players


    for i, session in enumerate(players):
        if session != players[-1] or i != len(players)-1:
            version = args.v_series
            if 'Bz' in version:
                q_scale = 1.2
            else:
                q_scale = 1.0
            SLM = model_utils.Network_Pcard_V2_2_BN_dropout(15+7, 7, y=1, x=15, lstmsize=args.ms_par0, hiddensize=args.ms_par1)
            QV = model_utils.Network_Qv_Universal_V1_2_BN_dropout(11*15,args.ms_par0,args.ms_par2,0.0,q_scale)

            SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{version}_{session}.pt')))
            QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{version}_{session}.pt')))
        
        else:
            version = args.v_gate
            if 'Bz' in version:
                q_scale = 1.2
            else:
                q_scale = 1.0
            SLM = model_utils.Network_Pcard_V2_2_BN_dropout(15+7, 7, y=1, x=15, lstmsize=args.mg_par0, hiddensize=args.mg_par1)
            QV = model_utils.Network_Qv_Universal_V1_2_BN_dropout(11*15,args.mg_par0,args.mg_par2,0.0,q_scale)

            SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{version}_{session}.pt')))
            QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{version}_{session}.pt')))
            
        
        SLM.eval()
        QV.eval()

        models.append(
            [SLM,QV]
            )
    
    ax_player = list(range(nplayer))

    outfile2 = os.path.join(wd,'data_gating',f'winrates_{"".join(versions)}_{fullplayers[0]}-{fullplayers[-2]}-{fullplayers[-1]}.txt')

    try:
        f = open(outfile2,'r').readlines()
        Fullstat = np.zeros((nplayer,3),dtype=np.int64)
        Fullstat[:,0] = np.array([int(s.split(' - ')[-3]) for s in f[1:]],dtype=np.int64)
        Fullstat[:,1] = np.array([int(s.split(' - ')[-2]) for s in f[1:]],dtype=np.int64)
        Fullstat[:,-1] = np.array([int(s.split(' - ')[-1]) for s in f[1:]],dtype=np.int64)
        #ngames = np.array([int(s.split('-')[-1]) for s in f[1:]],dtype=np.int64)
        csim = int(f[0].split()[1])
    except:
        csim = 0
        Fullstat = np.zeros((nplayer,3),dtype=np.int64)
        pass

    print(csim)
    #print(Fullstat)
    #quit()
    # random seed
    seed = random.randint(-1000000000,1000000000)
    print(seed)

    # create gating test
    pargs = []
    for i in range(len(players)):
        if i != gate_player_index:
            pargs.append((models[i], models[gate_player_index], 0,eval_device,15,64,nsim_perplayer,seed))
            pargs.append((models[gate_player_index], models[i], 0,eval_device,15,64,nsim_perplayer,seed))

    print(len(pargs))#,len(pargs[0]),len(models))
    # Create a pool of workers and distribute the tasks
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(simulate_match_wrapper, pargs, chunksize=1)
        #results = list(tqdm(pool.imap_unordered(simulate_match_wrapper, tasks), total=len(tasks)))

    #print(len(results))
    results = np.array(results).reshape(-1,2)
    results[:,1] = nsim_perplayer - results[:,1]
    #print(results)
    #quit()
    laststat = np.sum(nsim_perplayer-results,axis=0)[::-1]
    fullstat = np.append(results, laststat[None,:],axis=0)

    # wingame stats
    f = open(outfile2,'w')
    f.write(f'Nsim {int(csim+nsim)}\n')

    ngames = np.zeros(len(results)+1).astype(np.int64)+nsim
    ngames[-1] = (nsim)*(len(results))
    fullstat = np.append(fullstat,ngames[:,None],axis=1)
    if len(versions) > 1:
        print(fullstat)
        quit()
    Fullstat += fullstat

    for i in range(nplayer):
        if i != len(players)-1:
            version = args.v_series
        else:
            version = args.v_gate
        f.write(f'V{version}M{str(fullplayers[i]).zfill(3)} - {Fullstat[i][0]} - {Fullstat[i][1]} - {Fullstat[i][2]}\n')
    f.close()

    WRL = Fullstat[:,0] / Fullstat[:,-1]
    WRF = Fullstat[:,1] / Fullstat[:,-1]


    plt.figure(figsize=(10,4))
    plt.title(f'Nsim {int(csim+nsim)}')
    for i,v in enumerate(versions):
        plt.plot(fullplayers[i*len(players):(i+1)*len(players)],WRL)
        plt.plot(fullplayers[i*len(players):(i+1)*len(players)],WRF)
        
        total_win = np.int64((WRL+WRF)*int(csim+nsim))
        total_game = np.ones_like(total_win)*int(csim+nsim)*2
        #print(total_win, total_game)

        wr, wrint = calculate_win_rates_and_cis(total_win, total_game, 0.95)
        #print(wr,wrint)
        for x in range(len(WRF)):
            plt.text(fullplayers[i*len(players):(i+1)*len(players)][x],wr[x]/100,f'{round(wr[x],2)}\n{round(wrint[x],2)}')
        
        plt.errorbar(fullplayers[i*len(players):(i+1)*len(players)],(WRL+WRF)/2, yerr = np.array(wrint)/100)

    plt.axhline(0.5,zorder=-10,alpha=0.6,color='black')
    plt.gca().xaxis.set_tick_params(rotation=45)
    plt.grid()
    plt.savefig(os.path.join(wd,'data_gating',f'winrates_{"".join(versions)}_{fullplayers[0]}-{fullplayers[-2]}-{fullplayers[-1]}.png'),bbox_inches='tight')
    #plt.show()