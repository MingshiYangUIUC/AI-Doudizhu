import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from base_funcs_V2 import *
from model_V2 import *
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
    models, rand_param, selfplay_device, N_history, chunksize, ip, npr = task_params

    results = []

    results = simEpisode_batchpool_softmax(models, rand_param, selfplay_device, N_history, 64, chunksize.item())
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
    N_feature = 4
    version = f'H{str(N_history).zfill(2)}-V2_0.0'

    try:
        if sys.argv[1] == '-a': # auto detect
            mfiles = [int(f[-13:-3]) for f in os.listdir(os.path.join(wd,'models')) if version in f]
            if len(mfiles) == 0:
                Total_episodes = 0
            else:
                Total_episodes = max(mfiles)
            Add_episodes = int(sys.argv[2])
            Max_episodes = Total_episodes + Add_episodes
            #quit()
    except: # manual entry
        Total_episodes = 0 # current episode / model version
        Max_episodes   = 10000 # episode to stop. multiple of 50000
    
    print('B',Total_episodes, 'E',Max_episodes)
    
    N_episodes     = 12500 # number of games to be played before each round of training

    Save_frequency = 50000

    nprocess = 16 # number of process in selfplay

    # transfer weights from another version (need same model structure)
    transfer = False
    transfer_name = ''

    batch_size = 128
    nepoch = 1
    LR = 0.0001
    l2_reg_strength = 1e-6
    rand_param = 0.03 # "epsilon":" chance of pure random move; or "temperature" in the softmax version

    n_past_ds = 10 # augment using since last NP datasets.
    n_choice_ds = 1 # choose NC from last NP datasets

    n_worker = 0 # default dataloader

    #LM, DM, UM = Network_Pcard(N_history+N_feature),Network_Pcard(N_history+N_feature),Network_Pcard(N_history+N_feature)
    SLM = Network_Pcard(N_history+N_feature)
    QV = Network_Qv_Universal(5)
    #print(LM.nhist)
    print('Init wt',SLM.fc2.weight.data[0].mean())

    if Total_episodes > 0:
        SLM.load_state_dict(torch.load(os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt')))
        #DM.load_state_dict(torch.load(os.path.join(wd,'models',f'DM_{version}_{str(Total_episodes).zfill(10)}.pt')))
        #UM.load_state_dict(torch.load(os.path.join(wd,'models',f'UM_{version}_{str(Total_episodes).zfill(10)}.pt')))
        QV.load_state_dict(torch.load(os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt')))

    print('Init wt',SLM.fc2.weight.data[0].mean())

    if Total_episodes == 0:
        torch.save(SLM.state_dict(),os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt'))
        #torch.save(DM.state_dict(),os.path.join(wd,'models',f'DM_{version}_{str(Total_episodes).zfill(10)}.pt'))
        #torch.save(UM.state_dict(),os.path.join(wd,'models',f'UM_{version}_{str(Total_episodes).zfill(10)}.pt'))
        torch.save(QV.state_dict(),os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt'))

    while Total_episodes < Max_episodes:
        Xbuffer = [[],[],[]]
        Ybuffer = [[],[],[]]
        SL_X = []
        SL_Y = []
        chunksizes = torch.diff(torch.torch.linspace(0,N_episodes,nprocess+1).type(torch.int64))

        #print(chunksizes)
        #quit()
        t0 = time.time()
        with mp.Pool(nprocess) as pool:
            print(version, 'Playing games')

            SLM.eval()
            #DM.eval()
            #UM.eval()
            QV.eval()

            tasks = [([SLM, QV], rand_param, selfplay_device, N_history, chunksizes[_], _, nprocess) for _ in range(nprocess)]

            results = list(pool.imap(worker, tasks))
        
        Total_episodes += N_episodes
        t1 = time.time()
        print(version, 'Played games:',Total_episodes, 'Elapsed time:', round(t1-t0))
        torch.cuda.empty_cache()
        # End selfplay, process results
        Lwin = 0
        for chunk, sl_x, sl_y in results:
            SL_X.append(sl_x)
            SL_Y.append(sl_y)
            for BufferStatesActs, BufferRewards, success in chunk:
                if success:
                    for i in range(3):
                        #print(BufferStatesActs[i].shape)
                        Xbuffer[i].extend(BufferStatesActs[i])
                        Ybuffer[i].extend(BufferRewards[i])
                        if i == 0:
                            #print(len(Xbuffer[0]),len(Ybuffer[0]))
                            Lwin += BufferRewards[i][0][0]
        
        del results, tasks, chunk

        SL_X = torch.cat(SL_X)
        SL_Y = torch.cat(SL_Y)

        print([len(_) for _ in Ybuffer],'L WR:',round((Lwin/N_episodes).item()*100,2))

        print(SL_X.shape,SL_Y.shape)

        # SL part
        train_dataset = TensorDataset(SL_X, SL_Y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(SLM.parameters(), lr=LR, weight_decay=l2_reg_strength)
        SLM = train_model('cuda', SLM, torch.nn.MSELoss(), train_loader, nepoch, opt)
        SLM.to(device)
        print('Sample wt',SLM.fc2.weight.data[0].mean())
        #quit()

        # QV part
        X_full = torch.cat([torch.concat(list(Xbuffer[i])) for i in range(3)])
        Y_full = torch.cat([torch.concat(list(Ybuffer[i])) for i in range(3)]).unsqueeze(1)
        print(X_full.shape,Y_full.shape)

        train_dataset = TensorDataset(X_full, Y_full)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(QV.parameters(), lr=LR, weight_decay=l2_reg_strength)
        QV = train_model('cuda', QV, torch.nn.MSELoss(), train_loader, nepoch, opt)
        QV.to(device)

        gc.collect()
        torch.cuda.empty_cache()

        # save once every X episodes
        if Total_episodes % Save_frequency == 0:
            torch.save(SLM.state_dict(),os.path.join(wd,'models',f'SLM_{version}_{str(Total_episodes).zfill(10)}.pt'))
            torch.save(QV.state_dict(),os.path.join(wd,'models',f'QV_{version}_{str(Total_episodes).zfill(10)}.pt'))


'''
e:
conda activate rl-0
python E:\\Documents\\ddz\\train_v2.py
'''
