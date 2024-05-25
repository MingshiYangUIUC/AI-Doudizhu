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

def train_model(device, model, criterion, loader, nep, optimizer):
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(nep):
        running_loss = 0.0
        
        for inputs, targets in tqdm(loader):
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


#@profile
def worker(task_params):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    models, rand_param, selfplay_device, N_history, chunksize, ip, npr, bchance = task_params

    results = []

    results = simEpisode_batchpool_softmax(models, rand_param, selfplay_device, N_history, 32, chunksize.item(), bchance)
    torch.cuda.empty_cache()
    print(f'---------- {str(ip).zfill(2)} / {npr} END----------',end='\r')
    return results

def sampledataset(Total_episodes,N_episodes,N_back,label='L',kk=2):
    ds = []
    if N_back > 0:
        sample = [str(i).zfill(10) for i in list(range(Total_episodes-N_back*N_episodes,Total_episodes,N_episodes))]
        #print(sample)
        sampled = random.sample(sample,k=kk)
        print(sampled)
        
        for s in sampled:
            try:
                #print(s)
                ds.append(torch.load(os.path.join(wd,'train',f'{label}train_{version}_{s}_{N_episodes}')))
            except:
                pass
        #print(len(ds))
    return ds


if __name__ == '__main__':
    wd = os.path.dirname(__file__)
    mp.set_start_method('spawn', force=True)

    device = 'cpu'
    selfplay_device = 'cuda'

    N_history = 15 # number of historic moves in model input
    N_feature = 7
    version = f'H{str(N_history).zfill(2)}-V2_2.2'
    #version = f'H{str(N_history).zfill(2)}-V2_1.1'

    term = False
    try:
        if sys.argv[1] == '-a': # auto detect
            mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if version in f]
            if len(mfiles) == 0:
                Total_episodes = 0
                migrate=False
            else:
                Total_episodes = max(mfiles)
                migrate=True
            Add_episodes = int(sys.argv[2])
            Max_episodes = Total_episodes + Add_episodes
        elif sys.argv[1] == '-q': # query status
            mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if version in f]
            if len(mfiles) == 0:
                Total_episodes = 0
                migrate=False
            else:
                Total_episodes = max(mfiles)
                migrate=True
            Max_episodes = Total_episodes
            print('Current Progress:', Total_episodes)
            term = True
            #quit()
    except: # manual entry
        Total_episodes = 0 # current episode / model version
        Max_episodes   = 10000 # episode to stop. multiple of 50000
        migrate=False
    
    if term:
        sys.exit(0)

    print('B',Total_episodes, 'E',Max_episodes)
    
    N_episodes     = 25000 # number of games to be played before each round of training

    Save_frequency = 50000

    nprocess = 14 # number of process in selfplay

    # transfer weights from another version (need same model structure)
    transfer = False
    transfer_name = ''

    batch_size = 64
    nepoch = 1
    LR = 0.000003
    l2_reg_strength = 1e-6

    rand_param = 0.02 # "epsilon":" chance of pure random move; or "temperature" in the softmax version
    bomb_chance = 0.0 # chance of getting a deck full of bombs during selfplay

    n_past_ds = 10 # augment using since last NP datasets.
    n_choice_ds = 1 # choose NC from last NP datasets

    n_worker = 0 # default dataloader

    #LM, DM, UM = Network_Pcard(N_history+N_feature),Network_Pcard(N_history+N_feature),Network_Pcard(N_history+N_feature)
    SLM = Network_Pcard_V2_1(N_history+N_feature, N_feature, y=1, x=15, lstmsize=512, hiddensize=1024)
    QV = Network_Qv_Universal_V1_1(6,15,1024)

    #quit()
    #print(LM.nhist)
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
        chunksizes = torch.diff(torch.torch.linspace(0,N_episodes,nprocess*cs+1).type(torch.int64))

        t0 = time.time()
        with mp.Pool(nprocess) as pool:
            print(version, 'Playing games')

            SLM.eval()
            QV.eval()

            tasks = [([SLM, QV], rand_param, selfplay_device, N_history, chunksizes[_], _, nprocess, bomb_chance) for _ in range(nprocess*cs)]

            results = list(pool.imap_unordered(worker, tasks, chunksize=cs))
        
        Total_episodes += N_episodes
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
        print('Game Stat:',np.round((STAT/N_episodes)*100,2))

        print(SL_X.shape,SL_Y.shape)

        # SL part
        train_dataset = TensorDataset(SL_X, SL_Y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(SLM.parameters(), lr=LR, weight_decay=l2_reg_strength)
        SLM = train_model('cuda', SLM, torch.nn.MSELoss(), train_loader, nepoch, opt)
        SLM.to(device)
        print('Sample wt SL',SLM.fc2.weight.data[0].mean().item())

        # QV part
        X_full = torch.cat([torch.concat(list(Xbuffer[i])) for i in range(3)])
        Y_full = torch.cat([torch.concat(list(Ybuffer[i])) for i in range(3)]).unsqueeze(1)
        print(X_full.shape,Y_full.shape)

        train_dataset = TensorDataset(X_full, Y_full)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(QV.parameters(), lr=LR, weight_decay=l2_reg_strength)
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
python E:\\Documents\\ddz\\train_V2.py
'''
