import torch
import random
from collections import Counter
from itertools import combinations
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from base_funcs import *
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

def train_worker(Xbuffer_slice, Ybuffer_slice, model, batch_size, n_worker, LR, l2_reg_strength, nepoch, device):
    train_dataset = TensorDataset(torch.concat(Xbuffer_slice), torch.concat(Ybuffer_slice).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=l2_reg_strength)
    model = train_model(device, model, nn.MSELoss(), train_loader, nepoch, opt)
    
    del train_dataset, train_loader
    return model

#@profile
def worker(task_params):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    models, rand_param, selfplay_device, N_history, chunksize, ip, npr = task_params

    #models[0].to(selfplay_device)
    #models[1].to(selfplay_device)
    #models[2].to(selfplay_device)

    results = []

    '''
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(chunksize):
            percentage = int((_+1)/chunksize*100)
            if percentage > 0:
                eta = (time.perf_counter()-t0)/percentage*100
                print('-'*percentage+'_'*(100-percentage) + f' ETA: {round(eta)}  ',end='\r')
            result = simEpisode_softmax(models, rand_param, selfplay_device, N_history)
            results.append(result)
    '''
    '''
    ngame = 100
    clist = [ngame for i in range(0,chunksize,ngame)]
    clist[-1] = chunksize%ngame
    with torch.no_grad():
        if ip == npr-1:
            for c in tqdm(clist, desc="Processing batches"):
                torch.cuda.empty_cache()
                results += simEpisode_batch_softmax(models, rand_param, selfplay_device, N_history, c)
                #print(len(results), chunksize, '    ', end='\r')
        else:
            for c in clist:
                torch.cuda.empty_cache()
                results += simEpisode_batch_softmax(models, rand_param, selfplay_device, N_history, c)
                #print(len(results), chunksize, '    ', end='\r')
    '''
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

    LM, DM, UM = Network_V3(N_history+N_feature),Network_V3(N_history+N_feature),Network_V3(N_history+N_feature)
    #print(LM.nhist)
    print('Init wt',LM.fc2.weight.data[0].mean())

    Total_episodes = 0 # current episode / model version
    Max_episodes   = 1000 # episode to stop. multiple of 50000
    N_episodes     = 1000 # number of games to be played before each round of training

    Save_frequency = 50000

    nprocess = 16 # number of process in selfplay

    # transfer weights from another version (need same model structure)
    transfer = False
    transfer_name = ''

    version = f'H{str(N_history).zfill(2)}-V5'

    batch_size = 64
    nepoch = 1
    LR = 0.00003
    l2_reg_strength = 0
    rand_param = 1 # "epsilon":" chance of pure random move; or "temperature" in the softmax version

    n_past_ds = 0 # augment using since last NP datasets.
    n_choice_ds = 0 # choose NC from last NP datasets

    n_worker = 0 # default dataloader

    if transfer:
        print(f'use transfer {transfer_name}')
        LM.load_state_dict(torch.load(os.path.join(wd,'models',f'LM_{transfer_name}.pt')))
        DM.load_state_dict(torch.load(os.path.join(wd,'models',f'DM_{transfer_name}.pt')))
        UM.load_state_dict(torch.load(os.path.join(wd,'models',f'UM_{transfer_name}.pt')))

    elif Total_episodes > 0:
        LM.load_state_dict(torch.load(os.path.join(wd,'models',f'LM_{version}_{str(Total_episodes).zfill(10)}.pt')))
        DM.load_state_dict(torch.load(os.path.join(wd,'models',f'DM_{version}_{str(Total_episodes).zfill(10)}.pt')))
        UM.load_state_dict(torch.load(os.path.join(wd,'models',f'UM_{version}_{str(Total_episodes).zfill(10)}.pt')))

    print('Init wt',LM.fc2.weight.data[0].mean())

    if Total_episodes == 0:
        torch.save(LM.state_dict(),os.path.join(wd,'models',f'LM_{version}_{str(Total_episodes).zfill(10)}.pt'))
        torch.save(DM.state_dict(),os.path.join(wd,'models',f'DM_{version}_{str(Total_episodes).zfill(10)}.pt'))
        torch.save(UM.state_dict(),os.path.join(wd,'models',f'UM_{version}_{str(Total_episodes).zfill(10)}.pt'))

    while Total_episodes < Max_episodes:
        Xbuffer = [[],[],[]]
        Ybuffer = [[],[],[]]
        chunksizes = torch.diff(torch.torch.linspace(0,N_episodes,nprocess+1).type(torch.int64))

        #print(chunksizes)
        #quit()
        t0 = time.time()
        with mp.Pool(nprocess) as pool:
            print(version, 'Playing games')

            LM.eval()
            DM.eval()
            UM.eval()

            tasks = [([LM, DM, UM], rand_param, selfplay_device, N_history, chunksizes[_], _, nprocess) for _ in range(nprocess)]

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
        print(torch.concat(list(Ybuffer[0])).shape)
        quit()
        # train landlord
        train_dataset = TensorDataset(torch.concat(list(Xbuffer[0])), torch.concat(list(Ybuffer[0])).unsqueeze(1))
        torch.save(train_dataset,os.path.join(wd,'train',f'Ltrain_{str(Total_episodes-N_episodes).zfill(10)}_{N_episodes}'))

        # sample 2 from last 5
        ds = sampledataset(Total_episodes-N_episodes,N_episodes,n_past_ds,'L',n_choice_ds)
        print(f'Using previous {len(ds)} datasets')
        ds.append(train_dataset)
        train_dataset = ConcatDataset(ds)
        del ds
        # remove last 11+
        try:
            os.remove(os.path.join(wd,'train',f'Ltrain_{str(Total_episodes-n_past_ds+1*N_episodes).zfill(10)}_{N_episodes}'))
        except:pass
    
        print(len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(LM.parameters(), lr=LR, weight_decay=l2_reg_strength)
        LM = train_model('cuda', LM, torch.nn.MSELoss(), train_loader, nepoch, opt)
        LM.to(device)
        print('Sample wt',LM.fc2.weight.data[0].mean())

        del train_dataset, train_loader

        # train farmer 0
        train_dataset = TensorDataset(torch.concat(list(Xbuffer[1])), torch.concat(list(Ybuffer[1])).unsqueeze(1))
        torch.save(train_dataset,os.path.join(wd,'train',f'Dtrain_{str(Total_episodes-N_episodes).zfill(10)}_{N_episodes}'))

        # sample 2 from last 5
        ds = sampledataset(Total_episodes-N_episodes,N_episodes,n_past_ds,'D',n_choice_ds)
        print(f'Using previous {len(ds)} datasets')
        ds.append(train_dataset)
        train_dataset = ConcatDataset(ds)
        del ds
        # remove last 11+
        try:
            os.remove(os.path.join(wd,'train',f'Dtrain_{str(Total_episodes-n_past_ds+1*N_episodes).zfill(10)}_{N_episodes}'))
        except:pass

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(DM.parameters(), lr=LR, weight_decay=l2_reg_strength)
        DM = train_model('cuda', DM, torch.nn.MSELoss(), train_loader, nepoch, opt)
        DM.to(device)

        del train_dataset, train_loader

        # train farmer 1
        train_dataset = TensorDataset(torch.concat(list(Xbuffer[2])), torch.concat(list(Ybuffer[2])).unsqueeze(1))
        torch.save(train_dataset,os.path.join(wd,'train',f'Utrain_{str(Total_episodes-N_episodes).zfill(10)}_{N_episodes}'))

        # sample 2 from last 5
        ds = sampledataset(Total_episodes-N_episodes,N_episodes,n_past_ds,'U',n_choice_ds)
        print(f'Using previous {len(ds)} datasets')
        ds.append(train_dataset)
        train_dataset = ConcatDataset(ds)
        del ds
        # remove last 11+
        try:
            os.remove(os.path.join(wd,'train',f'Utrain_{str(Total_episodes-n_past_ds+1*N_episodes).zfill(10)}_{N_episodes}'))
        except:pass

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
        opt = torch.optim.Adam(UM.parameters(), lr=LR, weight_decay=l2_reg_strength)
        UM = train_model('cuda', UM, torch.nn.MSELoss(), train_loader, nepoch, opt)
        UM.to(device)

        del train_dataset, train_loader, Xbuffer, Ybuffer
        
        gc.collect()
        torch.cuda.empty_cache()

        # save once every X episodes
        if Total_episodes % Save_frequency == 0:
            torch.save(LM.state_dict(),os.path.join(wd,'models',f'LM_{version}_{str(Total_episodes).zfill(10)}.pt'))
            torch.save(DM.state_dict(),os.path.join(wd,'models',f'DM_{version}_{str(Total_episodes).zfill(10)}.pt'))
            torch.save(UM.state_dict(),os.path.join(wd,'models',f'UM_{version}_{str(Total_episodes).zfill(10)}.pt'))


'''
e:
conda activate rl-0
python E:\\Documents\\ddz\\train_v1.py
'''
