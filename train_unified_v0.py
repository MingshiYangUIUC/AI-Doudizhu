import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.nn.functional import one_hot

from base_funcs_unified import *
from model import *
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
        
        for inputs, player_ids, targets in tqdm(loader):
            inputs, player_ids, targets = inputs.to(device), player_ids.to(device), targets.to(device)
            
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs, player_ids)  # Forward pass, now includes player_ids
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
    model, rand_param, selfplay_device, N_history, chunksize, ip, npr = task_params

    results = simEpisode_batchpool_softmax(model, rand_param, selfplay_device, N_history, 32, chunksize.item())
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
                ds.append(torch.load(os.path.join(wd,'train',f'{label}train_{s}_{N_episodes}')))
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
    N_feature = 5

    Unified_model = Network_V3_Unified(N_history+N_feature)
    
    #print(LM.nhist)
    print('Init wt',Unified_model.fc2.weight.data[0].mean())

    Total_episodes = 1000000 # current episode / model version
    Max_episodes   = 1500000 # episode to stop. multiple of 50000
    N_episodes     = 10000 # number of games to be played before each round of training

    Save_frequency = 50000

    nprocess = 16 # number of process in selfplay

    # transfer weights from another version (need same model structure)
    transfer = False
    transfer_name = ''

    version = f'H{str(N_history).zfill(2)}-V5.1'

    batch_size = 64
    nepoch = 1
    LR = 0.00003
    l2_reg_strength = 0
    rand_param = 0.01 # "epsilon":" chance of pure random move; or "temperature" in the softmax version

    n_past_ds = 10 # augment using since last NP datasets.
    n_choice_ds = 0 # choose NC from last NP datasets

    n_worker = 0 # default dataloader

    if Total_episodes > 0:
        Unified_model.load_state_dict(torch.load(os.path.join(wd,'models',f'Unified_{version}_{str(Total_episodes).zfill(10)}.pt')))
        print('Init wt',Unified_model.fc2.weight.data[0].mean())

    elif Total_episodes == 0:
        torch.save(Unified_model.state_dict(),os.path.join(wd,'models',f'Unified_{version}_{str(Total_episodes).zfill(10)}.pt'))

    while Total_episodes < Max_episodes:
        T0 = time.time()

        Xbuffer = [[],[],[]]
        Ybuffer = [[],[],[]]
        chunksizes = torch.diff(torch.torch.linspace(0,N_episodes,nprocess+1).type(torch.int64))

        #print(chunksizes)
        #quit()
        t0 = time.time()
        with mp.Pool(nprocess) as pool:
            print(version, 'Playing games')

            Unified_model.eval()

            tasks = [(Unified_model, rand_param, selfplay_device, N_history, chunksizes[_], _, nprocess) for _ in range(nprocess)]

            results = list(pool.imap(worker, tasks))
        
        Total_episodes += N_episodes
        t1 = time.time()
        print(version, 'Played games:',Total_episodes, 'Elapsed time:', round(t1-t0))
        torch.cuda.empty_cache()
        # End selfplay, process results
        Lwin = 0
        for chunk in results:
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

        print([len(_) for _ in Ybuffer],'L WR:',round((Lwin/N_episodes).item()*100,2))

        # gather whole dataset

        Y_data = [torch.concat(list(Ybuffer[i])) for i in range(3)]
        ds_shape = [Y.shape[0] for Y in Y_data]
        X_data = [torch.concat(list(Xbuffer[i])) for i in range(3)]
        player_ids = torch.zeros(sum(ds_shape), 3)

        # Set the appropriate rows for each player
        player_ids[:ds_shape[0], 0] = 1  # Set the first A rows for player 0
        player_ids[ds_shape[0]:ds_shape[0]+ds_shape[1], 1] = 1  # Set the next B rows for player 1
        player_ids[ds_shape[0]+ds_shape[1]:, 2] = 1  # Set the last C rows for player 2

        train_dataset = TensorDataset(torch.concat(X_data), player_ids, torch.concat(Y_data).unsqueeze(1))
        torch.save(train_dataset,os.path.join(wd,'train',f'Unified_train_{str(Total_episodes-N_episodes).zfill(10)}_{N_episodes}'))
        
        ds = sampledataset(Total_episodes-N_episodes,N_episodes,n_past_ds,'Unified_',n_choice_ds)
        print(f'Using previous {len(ds)} datasets')
        ds.append(train_dataset)
        train_dataset = ConcatDataset(ds)
        
        del ds
        # remove last 11+
        try:
            os.remove(os.path.join(wd,'train',f'Ltrain_{str(Total_episodes-(n_past_ds+1)*N_episodes).zfill(10)}_{N_episodes}'))
        except:pass

        print(len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(Unified_model.parameters(), lr=LR, weight_decay=l2_reg_strength)
        Unified_model = train_model('cuda', Unified_model, torch.nn.MSELoss(), train_loader, nepoch, opt)
        Unified_model.to(device)
        print('Sample wt',Unified_model.fc1.weight.data[0].mean())

        del train_loader, train_dataset, player_ids, ds_shape, X_data, Y_data, Xbuffer, Ybuffer
        
        gc.collect()
        torch.cuda.empty_cache()

        # save once every X episodes
        if Total_episodes % Save_frequency == 0:
            torch.save(Unified_model.state_dict(),os.path.join(wd,'models',f'Unified_{version}_{str(Total_episodes).zfill(10)}.pt'))

        print(f'Advance. Current Iteration Time: {round(time.time()-T0,1)} seconds.')

'''
e:
conda activate rl-0
python E:\\Documents\\ddz\\train_unified_v0.py
'''
