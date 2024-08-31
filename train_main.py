"""

Script used to train models. 

Settings about training should be defined in 'TRAIN' section of 'config.ini'.

"""

import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import numpy as np
import time
import model_utils
import base_funcs_selfplay

from collections import deque
import torch.multiprocessing as mp
from tqdm import tqdm

import os
import sys
import gc
import pickle
import argparse
import configparser
from datetime import datetime, timezone

def set_seed(seed):
    # Set the seed for generating random numbers
    torch.manual_seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for Python random module
    random.seed(seed)

def train_model_onehead(device, model, criterion, loader, nep, optimizer):
    #scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(nep):
        running_loss = 0.0
        
        for inputs, targets in tqdm(loader):
            if inputs.size(0) > 1:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                outputs = model(inputs)[0]  # Forward pass
                loss = criterion(outputs, targets)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize
                optimizer.zero_grad()  # Zero the gradients
                
                running_loss += loss.item()
        
        # Calculate the average loss per batch over the epoch
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{nep}, Training Loss: {epoch_loss:.4f}")
    return model, epoch_loss

def train_model_aux(device, model, aux_weight, criterion, loader, nep, optimizer):
    #scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(nep):
        running_loss = 0.0
        
        for inputs, targets, auxiliary_targets in tqdm(loader):
            if inputs.size(0) > 1:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                auxiliary_targets = auxiliary_targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()  # Zero the gradients
                
                # Forward pass
                outputs_q, outputs_aux = model(inputs)
                
                # Calculate losses
                loss_q = criterion(outputs_q, targets)  # Loss for main task
                loss_aux = criterion(outputs_aux, auxiliary_targets)  # Loss for auxiliary task
                
                # Combine the losses (you can adjust the weight of auxiliary loss)
                loss = loss_q + aux_weight * loss_aux  # Adjust 0.1 as necessary for your use case
                
                # Backward pass
                loss.backward()
                
                # Optimize
                optimizer.step()
                
                running_loss += loss.item()
        
        # Calculate the average loss per batch over the epoch
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{nep}, Training Loss: {epoch_loss:.4f}")
    return model, epoch_loss

def train_model(device, model, criterion, loader, nep, optimizer):
    #scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(nep):
        running_loss = 0.0
        
        for inputs, targets in tqdm(loader):
            if inputs.size(0) > 1:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize
                optimizer.zero_grad()  # Zero the gradients
                
                running_loss += loss.item()
        
        # Calculate the average loss per batch over the epoch
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{nep}, Training Loss: {epoch_loss:.4f}")
    return model, epoch_loss

#@profile
def worker(task_params):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    #worker_seed = int((np.int32(time.time() * 1000) ** 5) * (os.getpid() ** 5)) % (2**32)
    worker_seed = random.randint(0,2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    print(f'---------- {os.getpid()} / {worker_seed} BEGIN----------',end='\r')

    models, rand_param, selfplay_device, selfplay_batch_size, n_history, chunksize, ip, npr, bchance, bs_max, ts_limit = task_params

    results = []
    if selfplay_device == 'cuda':
        torch.cuda.empty_cache()
    with torch.inference_mode():
        results = base_funcs_selfplay.simEpisode_batchpool_softmax(models, rand_param, selfplay_device, n_history, min(selfplay_batch_size,chunksize.item()), chunksize.item(), bchance, bs_max, ts_limit)
    if selfplay_device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    print(f'---------- {str(ip).zfill(2)} / {npr} END----------',end='\r')
    return results

def sampledataset(Total_episodes,n_episodes,N_back,label='L',kk=2):
    ds = []
    if N_back > 0:
        sample = [str(i).zfill(10) for i in list(range(Total_episodes-N_back*n_episodes,Total_episodes,n_episodes))]
        #print(sample)
        sampled = random.sample(sample,k=kk)
        print(sampled)
        
        for s in sampled:
            try:
                #print(s)
                ds.append(torch.load(os.path.join(wd,'train',f'{label}train_{version}_{s}_{n_episodes}')))
            except:
                pass
        #print(len(ds))
    return ds


def parse_args():
    parser = argparse.ArgumentParser(description="Run the training script with specified parameters.")
    
    # master arguments
    parser.add_argument('-a', '--auto', action='store_true', help="Auto train mode: continue iterating for task episode number")
    parser.add_argument('-t', '--train', type=int, default=10000, help="Task episode number")
    parser.add_argument('-q', '--query', action='store_true', help="Query status mode: return most recent model version")
    parser.add_argument('-l', '--logging', action='store_true', help="Log training status")
    parser.add_argument('-s', '--savedata', action='store_true', help="Store selfplay data to disk ./train")
    parser.add_argument('-f', '--freeze', type=str, default='', help="Freeze model (not train)?")

    # model arguments
    parser.add_argument('--version', type=str, default='V2_2.2', help="Version of the model")
    parser.add_argument('--n_history', type=int, default=15, help="Number of historic moves in model input")
    parser.add_argument('--n_feature', type=int, default=7, help="Additional feature size of the model")
    parser.add_argument('--m_par0', type=int, default=512, help="Model parameter 0: SLM LSTM")
    parser.add_argument('--m_par1', type=int, default=512, help="Model parameter 1: SLM MLP")
    parser.add_argument('--m_par2', type=int, default=512, help="Model parameter 2: QV MLP")
    parser.add_argument('--m_seed', type=int, default=20010101, help="Model init seed")

    # environment arguments
    parser.add_argument('--selfplay_device', type=str, default='cuda', help="Device for selfplay games")
    parser.add_argument('--n_save', type=int, default=100000, help="Number of games to be played before each round of saving")
    parser.add_argument('--n_refresh', type=int, default=100000, help="Number of games to be played before restarting mp pool")
    parser.add_argument('--n_processes', type=int, default=14, help="Number of CPU processes used in selfplay")
    parser.add_argument('--selfplay_batch_size', type=int, default=32, help="Batch number of concurrent games send to GPU by each process")
    parser.add_argument('--n_worker', type=int, default=0, help="num_workers in data loader")
    parser.add_argument('--max_inf_bs', type=int, default=30, help="Max true batch size send to model inference in each process")
    
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--n_episodes', type=int, default=25000, help="Number of games to be played before each round of training")
    parser.add_argument('--n_checkpoint', type=int, default=100000, help="Number of games to be played before each checkpoint model update")
    parser.add_argument('--timestep_limit', type=int, default=1000000, help="Limit this to prevent too many timesteps generated per training round")
    parser.add_argument('--n_epoch', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--lr1', type=float, default=0.000005/64*256, help="Scaled Learning rate for model part 1")
    parser.add_argument('--lr2', type=float, default=0.000003/64*256, help="Scaled Learning rate for model part 2")
    parser.add_argument('--l2_reg_strength', type=float, default=1e-6, help="L2 regularization strength")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--aux_weight', type=float, default=0.2, help="Auxiliary weight for expectation")
    parser.add_argument('--rand_param', type=float, default=0.01, help="Random parameter for action selection")
    parser.add_argument('--bomb_chance', type=float, default=0.01, help="Chance of getting a deck full of bombs during selfplay")
    
    # Add an argument for config file
    parser.add_argument('--config', type=str, default='.config.ini', help="Path to configuration file (relative)")

    args = parser.parse_args()

    # If config file is provided, parse it and update arguments
    if args.config:
        config = configparser.ConfigParser()
        wd = os.path.dirname(__file__)
        config.read(os.path.join(wd,args.config))

        # Update arguments from config file
        if 'TRAIN' in config:
            for key in vars(args):
                if key in config['TRAIN']:
                    config_value = config['TRAIN'][key]
                    if key == 'auto' or key == 'query' or key == 'logging' or key == 'savedata':
                        # Convert 'true'/'false' strings to boolean values
                        setattr(args, key, config_value.lower() == 'true')
                    else:
                        setattr(args, key, type(getattr(args, key))(config_value))

    return args

if __name__ == '__main__':
    wd = os.path.dirname(__file__)
    mp.set_start_method('spawn', force=True)
    #mp.set_sharing_strategy('file_system')

    if not os.path.isdir(os.path.join(wd,'logs')):
        os.mkdir(os.path.join(wd,'logs'))
    
    if not os.path.isdir(os.path.join(wd,'models')):
        os.mkdir(os.path.join(wd,'models'))

    if not os.path.isdir(os.path.join(wd,'train')):
        os.mkdir(os.path.join(wd,'train'))

    args = parse_args()

    device = 'cpu'
    selfplay_device = 'cuda'

    if args.savedata: # create directory if necessary
        if not os.path.isdir(os.path.join(wd,'train')):
            os.mkdir(os.path.join(wd,'train'))
            print('Created directory ./train to store data')

    selfplay_device = args.selfplay_device
    selfplay_batch_size = args.selfplay_batch_size
    n_history = args.n_history
    n_feature = args.n_feature
    version = f'H{str(n_history).zfill(2)}-{args.version}'

    term = False
    try:
        if args.auto:
            mfiles = [int(f.split('_')[-1][:-3]) for f in os.listdir(os.path.join(wd,'models')) if '_'+version+'_' in f]
            if len(mfiles) == 0:
                Total_episodes = 0
                migrate = False
            else:
                Total_episodes = max(mfiles)
                migrate = True
            Add_episodes = args.train
            Max_episodes = Total_episodes + Add_episodes
        elif args.query:
            mfiles = [int(f.split('_')[-1][:-3]) for f in os.listdir(os.path.join(wd,'models')) if '_'+version+'_' in f]
            if len(mfiles) == 0:
                Total_episodes = 0
                migrate = False
            else:
                Total_episodes = max(mfiles)
                migrate = True
            Max_episodes = Total_episodes
            print('Current Progress:', Total_episodes)
            term = True
    except:
        Total_episodes = 0
        Max_episodes = 10000
        migrate = False

    if term:
        sys.exit(0)

    print('B',Total_episodes, 'E',Max_episodes)
    
    freeze = args.freeze
    n_episodes = args.n_episodes
    Save_frequency = args.n_save
    n_process = args.n_processes
    batch_size = args.batch_size
    nepoch = args.n_epoch
    n_worker = args.n_worker
    bs_max = args.max_inf_bs
    ts_limit = args.timestep_limit
    LR1 = args.lr1 / 64 * batch_size
    LR2 = args.lr2 / 64 * batch_size
    l2_reg_strength = args.l2_reg_strength
    rand_param = args.rand_param
    bomb_chance = args.bomb_chance

    aux_weight = args.aux_weight

    # Initialize model
    random_seed = random.randint(0,2**32-1)
    if Total_episodes == 0:
        set_seed(args.m_seed)
    if args.logging and Total_episodes == 0:
        f = open(os.path.join(wd,'logs',f'training_stat_{version}.txt'),'w')
        f.write(f'Model init seed: {args.m_seed}\n')
        f.close()
    
    if 'Bz' in version:
        q_scale = 1.2
    else:
        q_scale = 1.0
    SLM = model_utils.Network_Pcard_V2_2_BN_dropout(n_history+n_feature, n_feature, y=1, x=15, lstmsize=args.m_par0, hiddensize=args.m_par1, dropout_rate=args.dropout)
    QV = model_utils.Network_Qv_Universal_V1_2_BN_dropout_auxiliary(input_size=(n_feature+1+2+1)*15,lstmsize=args.m_par0, hsize=args.m_par2, dropout_rate=args.dropout, scale_factor=q_scale, offset_factor=0.0) # action, lastmove, upper-lower state, action

    if (QV.scale != 1):
        print('Bz version.')

    # reset seed to random after initialization
    set_seed(random_seed)

    print('Init wt',SLM.fc2.weight.data[1].mean().item())

    # if you already have initialized version 0 models and saved them, use the saved version (migrate = True)
    migrate = os.path.isfile(os.path.join(wd,'models',f'SLM_{version}_{str(0).zfill(10)}.pt')) \
          and os.path.join(wd,'models',f'QV_{version}_{str(0).zfill(10)}.pt')
    
    if Total_episodes > 0 or migrate: # can migrate other model to be episode 0 model
        print('Migrated model parameters')
        SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt')))
        QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt')))

    print('Init wt',SLM.fc2.weight.data[1].mean().item())

    if Total_episodes == 0 and not migrate:
        torch.save(SLM.state_dict(),os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt'))
        torch.save(QV.state_dict(),os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt'))

    # create checkpoint
    SLM_ckpt = model_utils.Network_Pcard_V2_2_BN_dropout(n_history+n_feature, n_feature, y=1, x=15, lstmsize=args.m_par0, hiddensize=args.m_par1, dropout_rate=args.dropout)
    QV_ckpt = model_utils.Network_Qv_Universal_V1_2_BN_dropout_auxiliary(input_size=(n_feature+1+2+1)*15,lstmsize=args.m_par0, hsize=args.m_par2, dropout_rate=args.dropout, scale_factor=q_scale, offset_factor=0.0) # action, lastmove, upper-lower state, action

    SLM_ckpt.load_state_dict(SLM.state_dict())
    QV_ckpt.load_state_dict(QV.state_dict())

    pool = mp.Pool(n_process)

    next_eps = n_episodes

    while Total_episodes < Max_episodes:

        Xbuffer = [[],[],[]]
        Ybuffer = [[],[],[]]
        Yexp = [[],[],[]]
        SL_X = []
        SL_Y = []
        cs = 1
        
        nproc = min(n_process,next_eps)
        chunksizes = torch.diff(torch.torch.linspace(0,next_eps,nproc*cs+1).type(torch.int64))

        t0 = time.time()
        #with mp.Pool(n_process) as pool:
        
        print(version, 'Playing games')

        # selfplay using the checkpoint model - more stable
        SLM_ckpt.eval()
        QV_ckpt.eval()

        # mp
        tasks = [([SLM_ckpt, QV_ckpt], rand_param, selfplay_device, selfplay_batch_size, n_history, chunksizes[_], _, nproc, bomb_chance, bs_max, ts_limit//n_process) for _ in range(n_process*cs)]

        results = list(pool.imap_unordered(worker, tasks, chunksize=cs))
        # mp
        
        t1 = time.time()
        
        torch.cuda.empty_cache()
        
        # End selfplay, process results
        STAT = 0
        STAT = np.zeros(3)
        for SAs, Rewards, Exps, sl_x, sl_y, stat in results:
            SL_X.append(sl_x)
            SL_Y.append(sl_y)

            for i in range(3):
                Xbuffer[i].append(SAs[i])
                Ybuffer[i].append(Rewards[i])
                Yexp[i].append(Exps[i])
            
            STAT += stat
    
        del results, SAs, Rewards, Exps, sl_x, sl_y

        played_eps = int(sum(STAT))
        Total_episodes += played_eps
        print(version, 'Played games:',Total_episodes, 'Elapsed time:', round(t1-t0))
        print('Game Stat:',np.round((STAT/played_eps)*100,2))

        if played_eps == next_eps or (Total_episodes % n_episodes) == 0:
            next_eps = n_episodes
        else:
            next_eps = (n_episodes - Total_episodes) % n_episodes
            print(f'Reached timestep limit during this round. Next round only play {next_eps} games.')

        SL_X = torch.cat(SL_X)
        SL_Y = torch.cat(SL_Y)
        print(SL_X.shape,SL_Y.shape)
        #quit()
        # SL part
        if args.savedata: # save selfplay data
            note = f'{version}_{str(Total_episodes).zfill(10)}'
            torch.save(SL_X.clone().to(torch.int8), os.path.join(wd,'train',f'SL_X_int8_{note}.pt'))
            torch.save(SL_Y.clone().to(torch.int8), os.path.join(wd,'train',f'SL_Y_int8_{note}.pt'))

        if 'SL' not in freeze:
            #train_dataset = TensorDataset(SL_X.to('cuda'), SL_Y.to('cuda'))
            train_dataset = TensorDataset(SL_X, SL_Y)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
            opt = torch.optim.Adam(SLM.parameters(), lr=LR1, weight_decay=l2_reg_strength)
            SLM, slm_loss = train_model_onehead('cuda', SLM, torch.nn.MSELoss(), train_loader, nepoch, opt)
            SLM.to(device)
            print('Sample wt SL',SLM.fc2.weight.data[1].mean().item(),SLM_ckpt.fc2.weight.data[1].mean().item())
            del train_dataset, train_loader
        else:
            print('Skipped training due to freeze')
            slm_loss = -99999

        sy_size = SL_Y.shape
        del SL_X, SL_Y, 

        # QV part
        X_full = torch.cat([torch.concat(list(Xbuffer[i])) for i in range(3)])
        Y_full = torch.cat([torch.concat(list(Ybuffer[i])) for i in range(3)]).unsqueeze(1)
        Y_aux = torch.cat([torch.concat(list(Yexp[i])) for i in range(3)])
        print(X_full.shape, Y_full.shape, Y_aux.shape)

        if args.savedata: # save selfplay data
            note = f'{version}_{str(Total_episodes).zfill(10)}'
            # no need to save X_full, since X_full is constructed by [SL_X[:,:8,0], SL_y_hat, lstm_out_hat]
            # just save the action
            torch.save(X_full[:,-15:].clone().to(torch.int8), os.path.join(wd,'train',f'QV_X-action_int8_{note}.pt'))
            torch.save(Y_full.clone().to(torch.int8), os.path.join(wd,'train',f'QV_Y_int8_{note}.pt'))
        
        if 'QV' not in freeze:
            #train_dataset = TensorDataset(X_full.to('cuda'), Y_full.to('cuda'))
            train_dataset = TensorDataset(X_full, Y_full, Y_aux)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
            opt = torch.optim.Adam(QV.parameters(), lr=LR2, weight_decay=l2_reg_strength)
            QV, qv_loss = train_model_aux('cuda', QV, aux_weight, torch.nn.MSELoss(), train_loader, nepoch, opt)
            QV.to(device)
            print('Sample wt QV',QV.fc2.weight.data[0].mean().item(),QV_ckpt.fc2.weight.data[0].mean().item())
            del train_dataset, train_loader
        else:
            print('Skipped training due to freeze')
            qv_loss = -99999

        del X_full, Y_full, 

        gc.collect()
        torch.cuda.empty_cache()

        # create checkpoint every X episodes
        if Total_episodes % args.n_checkpoint == 0:
            SLM_ckpt.load_state_dict(SLM.state_dict())
            QV_ckpt.load_state_dict(QV.state_dict())
            print('Updated checkpoint models')

        # save once every X episodes
        if Total_episodes % Save_frequency == 0:
            torch.save(SLM_ckpt.state_dict(),os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt'))
            torch.save(QV_ckpt.state_dict(),os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt'))
            print('Saved models')
        
        # reset pool
        if Total_episodes % args.n_refresh == 0:
            if 'pool' in locals():
                pool.terminate()
                pool.join()
            pool = mp.Pool(n_process)
            print('Restarted pool')


        te = time.time()
        
        print('Episode time:', round(te-t0),'\n')

        if args.logging == True:
            # log status to file with name according to version
            f = open(os.path.join(wd,'logs',f'training_stat_{version}.txt'),'a')
            # episodes, game stats, train error, 
            f.write(f'Training phase done at UTC {datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")}\n')
            f.write(f'Simulated Episodes: {Total_episodes}; N rows: {sy_size[0]}\n')
            roundstat = np.round((STAT/played_eps)*100,4)
            f.write(f'Player status L F0 F1: {roundstat[0]}, {roundstat[1]}, {roundstat[2]}\n')
            f.write(f'Hyperparameters: batchsize={args.batch_size}, lr1={LR1}, lr2={LR2}, nep={args.n_epoch}, l2={args.l2_reg_strength}, dropout={args.dropout}, rand={args.rand_param}, bomb={args.bomb_chance}\n')
            f.write(f'Model losses SLM QV: {slm_loss}, {qv_loss}\n')
            f.close()

    if 'pool' in locals():
        pool.terminate()
        pool.join()
