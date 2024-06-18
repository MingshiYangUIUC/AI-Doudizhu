import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from model_utils import *
from base_utils import *
from base_funcs_selfplay import simEpisode_batchpool_softmax
#from test_batch_sim import *

from collections import deque
import torch.multiprocessing as mp
from tqdm import tqdm

import os
import sys
import gc
import pickle
import argparse

def train_model(device, model, criterion, loader, nep, optimizer):
    #scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(nep):
        running_loss = 0.0
        
        for inputs, targets in tqdm(loader):
            if inputs.size(0) > 1:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize
                
                running_loss += loss.item()
        
        # Calculate the average loss per batch over the epoch
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{nep}, Training Loss: {epoch_loss:.4f}")
    return model


def train_model_amp(device, model, criterion, loader, nep, optimizer):
    scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5)
    model.to(device)
    model.train()
    for epoch in range(nep):
        running_loss = 0.0
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{nep}, Training Loss: {epoch_loss:.4f}")
    return model


#@profile
def worker(task_params):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    models, rand_param, selfplay_device, selfplay_batch_size, n_history, chunksize, ip, npr, bchance = task_params

    results = []
    torch.cuda.empty_cache()
    results = simEpisode_batchpool_softmax(models, rand_param, selfplay_device, n_history, selfplay_batch_size, chunksize.item(), bchance)
    torch.cuda.empty_cache()
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

    # model arguments
    parser.add_argument('--version', type=str, default='V2_2.2', help="Version of the model")
    parser.add_argument('--n_history', type=int, default=15, help="Number of historic moves in model input")
    parser.add_argument('--n_feature', type=int, default=7, help="Additional feature size of the model")

    # environment arguments
    parser.add_argument('--selfplay_device', type=str, default='cuda', help="Device for selfplay games")
    parser.add_argument('--n_save', type=int, default=100000, help="Number of games to be played before each round of saving")
    parser.add_argument('--n_processes', type=int, default=14, help="Number of CPU processes used in selfplay")
    parser.add_argument('--selfplay_batch_size', type=int, default=32, help="Batch number of concurrent games send to GPU by each process")
    
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--n_episodes', type=int, default=25000, help="Number of games to be played before each round of training")
    parser.add_argument('--n_epoch', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--lr1', type=float, default=0.000005/64*256, help="Learning rate for model part 1")
    parser.add_argument('--lr2', type=float, default=0.000003/64*256, help="Learning rate for model part 2")
    parser.add_argument('--l2_reg_strength', type=float, default=1e-6, help="L2 regularization strength")
    parser.add_argument('--rand_param', type=float, default=0.01, help="Random parameter for moves")
    parser.add_argument('--bomb_chance', type=float, default=0.01, help="Chance of getting a deck full of bombs during selfplay")
    
    return parser.parse_args()

if __name__ == '__main__':
    wd = os.path.dirname(__file__)
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    device = 'cpu'
    selfplay_device = 'cuda'

    selfplay_device = args.selfplay_device
    selfplay_batch_size = args.selfplay_batch_size
    n_history = args.n_history
    n_feature = args.n_feature
    version = f'H{str(n_history).zfill(2)}-{args.version}'

    term = False
    try:
        if args.auto:
            mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if '_'+version+'_' in f]
            if len(mfiles) == 0:
                Total_episodes = 0
                migrate = False
            else:
                Total_episodes = max(mfiles)
                migrate = True
            Add_episodes = args.train
            Max_episodes = Total_episodes + Add_episodes
        elif args.query:
            mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if version in f]
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
    
    n_episodes = args.n_episodes
    Save_frequency = args.n_save
    n_process = args.n_processes
    batch_size = args.batch_size
    nepoch = args.n_epoch
    LR1 = args.lr1
    LR2 = args.lr2
    l2_reg_strength = args.l2_reg_strength
    rand_param = args.rand_param
    bomb_chance = args.bomb_chance


    # transfer weights from another version (need same model structure)
    #transfer = False
    #transfer_name = ''

    #n_past_ds = 10 # augment using since last NP datasets.
    #n_choice_ds = 1 # choose NC from last NP datasets

    n_worker = 0 # default dataloader

    #SLM = Network_Pcard_V2_1(n_history+n_feature, n_feature, y=1, x=15, lstmsize=512, hiddensize=1024)
    #QV = Network_Qv_Universal_V1_1(6,15,1024)

    SLM = Network_Pcard_V2_1_BN(n_history+n_feature, n_feature, y=1, x=15, lstmsize=512, hiddensize=512)
    QV = Network_Qv_Universal_V1_1_BN(6,15,512)
    #SLM = Network_Pcard_V2_1_Trans(n_history+n_feature, n_feature, y=1, trans_heads=4, trans_layers=6, hiddensize=512)
    #QV = Network_Qv_Universal_V1_1(6,15,512)

    print('Init wt',SLM.fc2.weight.data[0].mean().item())

    if Total_episodes > 0 or migrate: # can migrate other model to be episode 0 model
        SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt')))
        QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt')))

    print('Init wt',SLM.fc2.weight.data[0].mean().item())

    if Total_episodes == 0 and not migrate:
        torch.save(SLM.state_dict(),os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt'))
        torch.save(QV.state_dict(),os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt'))

    while Total_episodes < Max_episodes:

        Xbuffer = [[],[],[]]
        Ybuffer = [[],[],[]]
        SL_X = []
        SL_Y = []
        cs = 1
        chunksizes = torch.diff(torch.torch.linspace(0,n_episodes,n_process*cs+1).type(torch.int64))

        t0 = time.time()
        with mp.Pool(n_process) as pool:
            print(version, 'Playing games')

            SLM.eval()
            QV.eval()

            tasks = [([SLM, QV], rand_param, selfplay_device, selfplay_batch_size, n_history, chunksizes[_], _, n_process, bomb_chance) for _ in range(n_process*cs)]

            results = list(pool.imap_unordered(worker, tasks, chunksize=cs))
        
        Total_episodes += n_episodes
        t1 = time.time()
        print(version, 'Played games:',Total_episodes, 'Elapsed time:', round(t1-t0))
        torch.cuda.empty_cache()
        
        # End selfplay, process results
        STAT = 0
        STAT = np.zeros(3)
        for SAs, Rewards, sl_x, sl_y, stat in results:
            SL_X.append(sl_x)
            SL_Y.append(sl_y)

            for i in range(3):
                #print(BufferStatesActs[i].shape)
                Xbuffer[i].append(SAs[i])
                Ybuffer[i].append(Rewards[i])
                #print(SAs[i].shape, Rewards[i].shape)
                #print(BufferRewards[i][0], torch.zeros(len(BufferStatesActs[i][0]))+BufferRewards[i][0][0])
                #print(len(BufferStatesActs[i]),torch.zeros(len(BufferStatesActs[i]))+BufferRewards[i])
            
            STAT += stat
    
        del results, SAs, Rewards, sl_x, sl_y
        #del results, tasks, chunk
        

        SL_X = torch.cat(SL_X)
        SL_Y = torch.cat(SL_Y)

        #STAT[1:] = np.sum(STAT[1:])
        print('Game Stat:',np.round((STAT/n_episodes)*100,2))

        print(SL_X.shape,SL_Y.shape)

        # SL part
        train_dataset = TensorDataset(SL_X.to('cuda'), SL_Y.to('cuda'))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker)#, pin_memory=True)
        opt = torch.optim.Adam(SLM.parameters(), lr=LR1, weight_decay=l2_reg_strength)
        SLM = train_model('cuda', SLM, torch.nn.MSELoss(), train_loader, nepoch, opt)
        SLM.to(device)
        print('Sample wt SL',SLM.fc2.weight.data[0].mean().item())

        # QV part
        X_full = torch.cat([torch.concat(list(Xbuffer[i])) for i in range(3)])
        Y_full = torch.cat([torch.concat(list(Ybuffer[i])) for i in range(3)]).unsqueeze(1)
        print(X_full.shape,Y_full.shape)

        train_dataset = TensorDataset(X_full.to('cuda'), Y_full.to('cuda'))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker)#, pin_memory=True)
        opt = torch.optim.Adam(QV.parameters(), lr=LR2, weight_decay=l2_reg_strength)
        QV = train_model('cuda', QV, torch.nn.MSELoss(), train_loader, nepoch, opt)
        QV.to(device)
        print('Sample wt QV',QV.fc2.weight.data[0].mean().item())

        gc.collect()
        torch.cuda.empty_cache()

        # save once every X episodes
        if Total_episodes % Save_frequency == 0:
            torch.save(SLM.state_dict(),os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt'))
            torch.save(QV.state_dict(),os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt'))

        te = time.time()
        
        print('Episode time:', round(te-t0),'\n')

'''
e:
conda activate rl-0
python E:\\Documents\\ddz\\train_main.py
'''
