"""

Define pytorch neural network classes. Used by othe scripts, do not run alone.

There are some legacy networks not being used anymore.
There are helper functions that search and return an action based on given game state. Used by 'pvc.py' and 'ai_helper.py'.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_utils import avail_actions_cpp, str2state, r2c_base_arr, Label


# old models

class Network_Qv_Universal_V1_1(nn.Module): # this network uses estimated state to estimate q values of action
                               # use 3 states (SELF, UPPER, LOWER), 1 role, and 1 action (Nelems) (z=5)
                               # cound use more features such as history
                               # should be simpler

    def __init__(self, z, x=15, hsize=256):
        super(Network_Qv_Universal_V1_1, self).__init__()
        self.x = x
        self.z = z

        input_size = z * x
        hidden_size = hsize

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1) # output is q values
        self.flatten = nn.Flatten()

    def forward(self, x):

        # Process through FNN
        x = self.flatten(x)
        x = F.relu(self.fc1(x))

        x1 = x
        x = F.relu(self.fc2(x))
        x = x + x1

        x1 = x
        x = F.relu(self.fc3(x))
        x = x + x1

        x1 = x
        x = F.relu(self.fc4(x))
        x = x + x1

        x = torch.sigmoid(self.fc5(x))
        return x

class Network_Pcard_V2_1(nn.Module): # this network considers public cards of landlord
                                 # predict opponent states (as frequency of cards) (UPPER, LOWER)
                                 # does not read in action
    def __init__(self, z, nh=5, y=1, x=15, lstmsize=512, hiddensize=512):
        super(Network_Pcard_V2_1, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = nh # self, unavail, card count, visible, ROLE, NO action
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstmsize, batch_first=True)

        # Calculate the size of the non-historical input
        input_size = self.non_hist * y * x + lstmsize  # +lstmsize for the LSTM output
        hidden_size = hiddensize

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 2*x) # output is 2 states (Nelems)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Extract the historical layers for LSTM processing
        historical_data = x[:, self.non_hist:, :, :]
        historical_data = historical_data.flip(dims=[1])
        historical_data = historical_data.reshape(-1, self.nhist, self.y * self.x)  # Reshape for LSTM

        # Process historical layers with LSTM
        lstm_out, _ = self.lstm(historical_data)
        lstm_out = lstm_out[:, -1, :]  # Use only the last output of the LSTM

        # Extract and flatten the non-historical part of the input
        non_historical_data = x[:, :self.non_hist, :, :]
        non_historical_data = non_historical_data.reshape(-1, non_historical_data.shape[1] * self.y * self.x)

        # Concatenate LSTM output with non-historical data
        x = torch.cat((lstm_out, non_historical_data), dim=1)

        # Process through FNN as before
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x1 = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x + x1
        x1 = x
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = x + x1
        x = F.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))*4 # max count is 4, min count is 0
        return x


# V2_2_2 models

class Network_Pcard_V2_1_BN(nn.Module): # this network considers public cards of landlord
                                 # predict opponent states (as frequency of cards) (UPPER, LOWER)
                                 # does not read in action
    def __init__(self, z, nh=5, y=1, x=15, lstmsize=512, hiddensize=512):
        super(Network_Pcard_V2_1_BN, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = nh # self, unavail, card count, visible, ROLE, NO action
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstmsize, batch_first=True)

        # Calculate the size of the non-historical input
        input_size = self.non_hist * y * x + lstmsize  # +lstmsize for the LSTM output
        hidden_size = hiddensize

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 2*x) # output is 2 states (Nelems)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Extract the historical layers for LSTM processing
        historical_data = x[:, self.non_hist:, :, :]
        historical_data = historical_data.flip(dims=[1])
        historical_data = historical_data.reshape(-1, self.nhist, self.y * self.x)  # Reshape for LSTM

        # Process historical layers with LSTM
        lstm_out, _ = self.lstm(historical_data)
        lstm_out = lstm_out[:, -1, :]  # Use only the last output of the LSTM

        # Extract and flatten the non-historical part of the input
        non_historical_data = x[:, :self.non_hist, :, :]
        non_historical_data = non_historical_data.reshape(-1, non_historical_data.shape[1] * self.y * self.x)

        # Concatenate LSTM output with non-historical data
        x = torch.cat((lstm_out, non_historical_data), dim=1)

        # Process through FNN as before
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x1 = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = x + x1
        x1 = x
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = x + x1
        x = F.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))*4 # max count is 4, min count is 0
        return x


class Network_Qv_Universal_V1_1_BN(nn.Module): # this network uses estimated state to estimate q values of action
                               # use 3 states (SELF, UPPER, LOWER), 1 role, and 1 action (Nelems) (z=5)
                               # cound use more features such as history
                               # should be simpler

    def __init__(self, z, x=15, hsize=256):
        super(Network_Qv_Universal_V1_1_BN, self).__init__()
        self.x = x
        self.z = z

        input_size = z * x
        hidden_size = hsize

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1) # output is q values
        self.flatten = nn.Flatten()

    def forward(self, x):

        # Process through FNN
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))

        x1 = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = x + x1

        x1 = x
        x = F.relu(self.fc3(x))
        x = x + x1

        x1 = x
        x = F.relu(self.fc4(x))
        x = x + x1

        x = torch.sigmoid(self.fc5(x))
        return x


# V2_3_0 models

class Network_Pcard_V2_2_BN_dropout(nn.Module): # this network considers public cards of landlord
                                 # predict opponent states (as frequency of cards) (UPPER, LOWER)
                                 # does not read in action
                                 # output lstm encoding
    def __init__(self, z, nh=5, y=1, x=15, lstmsize=512, hiddensize=512, dropout_rate=0.5):
        super(Network_Pcard_V2_2_BN_dropout, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = nh # self, unavail, card count, visible, ROLE, NO action
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        self.dropout = nn.Dropout(dropout_rate)

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstmsize, batch_first=True)

        # Calculate the size of the non-historical input
        input_size = self.non_hist * y * x + lstmsize  # +lstmsize for the LSTM output
        hidden_size = hiddensize

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 2*x) # output is 2 states (Nelems)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Extract the historical layers for LSTM processing
        historical_data = x[:, self.non_hist:, :, :]
        historical_data = historical_data.flip(dims=[1])
        historical_data = historical_data.reshape(-1, self.nhist, self.y * self.x)  # Reshape for LSTM

        # Process historical layers with LSTM
        lstm_out, _ = self.lstm(historical_data)
        lstm_out = lstm_out[:, -1, :]  # Use only the last output of the LSTM

        # Extract and flatten the non-historical part of the input
        non_historical_data = x[:, :self.non_hist, :, :]
        non_historical_data = non_historical_data.reshape(-1, non_historical_data.shape[1] * self.y * self.x)

        # Concatenate LSTM output with non-historical data
        x = torch.cat((lstm_out, non_historical_data), dim=1)

        # Process through FNN as before
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x1 = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = x + x1
        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = x + x1
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))*4 # max count is 4, min count is 0
        return x, lstm_out


class Network_Qv_Universal_V1_2_BN_dropout(nn.Module): # this network uses estimated state to estimate q values of action
                               # use 3 states (SELF, UPPER, LOWER), 1 role, and 1 action (Nelems) (z=5)
                               # cound use more features such as history
                               # should be simpler
                               # use lstm and Bigstate altogether

    def __init__(self, input_size, lstmsize, hsize=256, dropout_rate=0.5, scale_factor=1.0, offset_factor=0.0):
        super(Network_Qv_Universal_V1_2_BN_dropout, self).__init__()

        hidden_size = hsize

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(input_size + lstmsize, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1) # output is q values
        self.flatten = nn.Flatten()
        self.scale = scale_factor
        self.offset = offset_factor

    def forward(self, x):

        # Process through FNN
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = x + x1

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc3(x))
        x = x + x1

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc4(x))
        x = x + x1

        x = torch.sigmoid(self.fc5(x))

        x = x * self.scale + self.offset
        return x


class Network_Qv_Universal_V1_2_BN_dropout_auxiliary(nn.Module): # this network uses estimated state to estimate q values of action
                               # use 3 states (SELF, UPPER, LOWER), 1 role, and 1 action (Nelems) (z=5)
                               # cound use more features such as history
                               # should be simpler
                               # use lstm and Bigstate altogether
                               # this network also predict opponent action, although not directly used.

    def __init__(self, input_size, lstmsize, hsize=256, dropout_rate=0.5, scale_factor=1.0, offset_factor=0.0):
        super(Network_Qv_Universal_V1_2_BN_dropout_auxiliary, self).__init__()

        hidden_size = hsize

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(input_size + lstmsize, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1) # output is q values
        self.flatten = nn.Flatten()
        self.scale = scale_factor
        self.offset = offset_factor

        # Auxiliary task (opponent action prediction) output layer
        self.auxiliary_output = nn.Linear(hidden_size, 45)

    def forward(self, x):

        # Process through FNN
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = x + x1

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc3(x))
        x = x + x1

        # Auxiliary output after fc3
        aux_output = torch.sigmoid(self.auxiliary_output(x))*4

        x = self.dropout(x)
        x1 = x
        x = F.relu(self.fc4(x))
        x = x + x1

        x = torch.sigmoid(self.fc5(x))

        x = x * self.scale + self.offset
        return x, aux_output


class ResBlock(nn.Module):
    def __init__(self, width):
        super(ResBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        residual = x  # Save the input for the residual connection
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x + residual  # Add the input (residual connection) to the output
    


class Network_Qv_V2_0_Resblock(nn.Module):
    def __init__(self, input_size, lstmsize, num_resblocks=3, hsize=256, dropout_rate=0.5, scale_factor=1.0, offset_factor=0.0):
        super(Network_Qv_V2_0_Resblock, self).__init__()

        self.hsize = hsize

        # Initial layer connecting input to the first residual block
        self.fc1 = nn.Linear(input_size + lstmsize, hsize)
        #self.bn1 = nn.BatchNorm1d(hsize)
        #self.dropout = nn.Dropout(dropout_rate)

        # Create the residual blocks
        self.resblocks = nn.ModuleList([ResBlock(hsize) for _ in range(num_resblocks)])

        # Final layers after the residual blocks
        self.fc_final = nn.Linear(hsize, 1)  # Output layer for Q values

        # Auxiliary task output layer
        self.auxiliary_output = nn.Linear(hsize, 45)

        self.flatten = nn.Flatten()
        self.scale = scale_factor
        self.offset = offset_factor

    def forward(self, x):
        # Process through the initial linear, batch norm, and dropout layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)

        # Process through the residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # Auxiliary output after residual blocks
        aux_output = torch.sigmoid(self.auxiliary_output(x)) * 4

        # Final output layer
        x = torch.sigmoid(self.fc_final(x))
        x = x * self.scale + self.offset
        return x, aux_output

class ResBlock2(nn.Module):
    def __init__(self, width):
        super(ResBlock2, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        residual = x  # Save the input for the residual connection
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)) + residual) # Add the input (residual connection) to the output
        return x  

class Network_Qv_V2_1_Resblock(nn.Module):
    def __init__(self, input_size, lstmsize, num_resblocks=3, hsize=256, dropout_rate=0.5, scale_factor=1.0, offset_factor=0.0):
        super(Network_Qv_V2_1_Resblock, self).__init__()

        self.hsize = hsize

        # Initial layer connecting input to the first residual block
        self.fc1 = nn.Linear(input_size + lstmsize, hsize)
        #self.bn1 = nn.BatchNorm1d(hsize)
        #self.dropout = nn.Dropout(dropout_rate)

        # Create the residual blocks
        self.resblocks = nn.ModuleList([ResBlock2(hsize) for _ in range(num_resblocks)])

        # Final layers after the residual blocks
        self.fc_final = nn.Linear(hsize, 1)  # Output layer for Q values

        # Auxiliary task output layer
        self.auxiliary_output = nn.Linear(hsize, 45)

        self.flatten = nn.Flatten()
        self.scale = scale_factor
        self.offset = offset_factor

    def forward(self, x):
        # Process through the initial linear, batch norm, and dropout layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)

        # Process through the residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # Auxiliary output after residual blocks
        aux_output = torch.sigmoid(self.auxiliary_output(x)) * 4

        # Final output layer
        x = torch.sigmoid(self.fc_final(x))
        x = x * self.scale + self.offset
        return x, aux_output

# wrapper functions for the models defined above

def get_action_serial_V2_2_2(Turn, SLM, QV, Initstates,unavail,lastmove, Forcemove, history, temperature, hint=False):
    player = Initstates[Turn%3]#.clone().detach()
    visible = Initstates[-1]#.clone().detach()

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((3,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            unavail.unsqueeze(0),
                            CC,
                            visible.unsqueeze(0), # new feature
                            torch.full((1, 15), Turn%3),
                            history])
    Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
    #print(Bigstate)
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    if hint:
        cprob = model_inter.numpy().round(1).reshape(2,15)

        print('Hint:      ','    '.join([f' {c} ' for c in r2c_base_arr]))

        c0 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in cprob[0]]
        c1 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in cprob[1]]

        print(f'{Label[(Turn-1)%3]}:  ','    '.join([str(c) for c in c0]))
        print(f'{Label[(Turn+1)%3]}:  ','    '.join([str(c) for c in c1]))

    # get all actions
    #print(player)
    acts = avail_actions_cpp(lastmove[0],lastmove[1],player,Forcemove)
    #print(acts)
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                hinput[:,7].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    #print(output)
    if temperature == 0:
        Q = torch.max(output)
        best_act = acts[torch.argmax(output)]
    else:
        # get action using probabilistic approach and temperature
        probabilities = torch.softmax(output / temperature, dim=0)
        distribution = torch.distributions.Categorical(probabilities)
        
        q = distribution.sample()
        best_act = acts[q]
        Q = output[q]
    
    action = best_act
    return action, Q

def evaluate_action_serial_V2_2_2(Turn, SLM, QV, Initstates, unavail, lastmove, Forcemove, history, temperature, Action):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((3,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            unavail.unsqueeze(0),
                            CC,
                            visible.unsqueeze(0), # new feature
                            torch.full((1, 15), Turn%3),
                            history])
    Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    acts = [Action]
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                hinput[:,7].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    Q = output
    return Q


def get_action_serial_V2_3_0(Turn, SLM, QV, Initstates, unavail, played_cards, lastmove, Forcemove, history, temperature, hint=False):
    player = Initstates[Turn%3]#.clone().detach()
    visible = Initstates[-1]#.clone().detach()

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            unavail.unsqueeze(0),
                            played_cards,
                            visible.unsqueeze(0), # new feature
                            torch.full((1, 15), Turn%3),
                            history])
    Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
    #print(Bigstate)
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    #print(hinput)
    model_inter, lstm_out = SLM(hinput)
    #role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    if hint:
        cprob = model_inter.numpy().round(1).reshape(2,15)

        print('Hint:      ','    '.join([f' {c} ' for c in r2c_base_arr]))

        c0 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in cprob[0]]
        c1 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in cprob[1]]

        print(f'{Label[(Turn-1)%3]}:  ','    '.join([str(c) for c in c0]))
        print(f'{Label[(Turn+1)%3]}:  ','    '.join([str(c) for c in c1]))

    # get all actions
    #print(player)
    acts = avail_actions_cpp(lastmove[0],lastmove[1],player,Forcemove)
    #print(acts)
    # generate inputs 2
    model_inter = torch.concat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                model_inter, # upper and lower states
                                #role,
                                lstm_out, # lstm encoded history
                                ],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    #print(output)
    if temperature == 0:
        Q = torch.max(output)
        best_act = acts[torch.argmax(output)]
    else:
        # get action using probabilistic approach and temperature
        probabilities = torch.softmax(output / temperature, dim=0)
        distribution = torch.distributions.Categorical(probabilities)
        
        q = distribution.sample()
        best_act = acts[q]
        Q = output[q]

    if hint: # get first few acts
        qs = torch.argsort(output).flip(dims=(0,))[:5]
        acts_sort = [acts[i] for i in qs]
        print('Candidates:',' | '.join([f'{"pass" if acts_sort[i][0] == "" else acts_sort[i][0]} {str(round(output[qs[i]].item()*100,1)).zfill(5)}%' for i in range(len(qs))]))
    
    action = best_act
    return action, Q


def evaluate_action_serial_V2_3_0(Turn, SLM, QV, Initstates, unavail, played_cards,lastmove, Forcemove, history, temperature, Action):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            unavail.unsqueeze(0),
                            played_cards,
                            visible.unsqueeze(0), # new feature
                            torch.full((1, 15), Turn%3),
                            history])
    Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
    #print(Bigstate)
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter, lstm_out = SLM(hinput)
    #role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    acts = [Action]
    # generate inputs 2
    model_inter = torch.concat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                model_inter, # upper and lower states
                                #role,
                                lstm_out, # lstm encoded history
                                ],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    Q = output
    return Q

def get_action_serial_V2_4_0(Turn, SLM, QV, Initstates, unavail, played_cards, lastmove, Forcemove, history, temperature, hint=False):
    player = Initstates[Turn%3]#.clone().detach()
    visible = Initstates[-1]#.clone().detach()

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            unavail.unsqueeze(0),
                            played_cards,
                            visible.unsqueeze(0), # new feature
                            torch.full((1, 15), Turn%3),
                            history])
    Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
    #print(Bigstate)
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    #print(hinput)
    model_inter, lstm_out = SLM(hinput)
    #role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    if hint:
        cprob = model_inter.numpy().round(1).reshape(2,15)

        print('Hint:      ','    '.join([f' {c} ' for c in r2c_base_arr]))

        c0 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in cprob[0]]
        c1 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in cprob[1]]

        print(f'{Label[(Turn-1)%3]}:  ','    '.join([str(c) for c in c0]))
        print(f'{Label[(Turn+1)%3]}:  ','    '.join([str(c) for c in c1]))

    # get all actions
    #print(player)
    acts = avail_actions_cpp(lastmove[0],lastmove[1],player,Forcemove)
    #print(acts)
    # generate inputs 2
    model_inter = torch.concat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                model_inter, # upper and lower states
                                #role,
                                lstm_out, # lstm encoded history
                                ],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a[0]).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2)
    expect = output[1]
    #print(expect.shape)
    output = output[0].flatten()
    #print(output)
    if temperature == 0:
        Q = torch.max(output)
        best_act = acts[torch.argmax(output)]
        best_exp = expect[torch.argmax(output)]
    else:
        # get action using probabilistic approach and temperature
        probabilities = torch.softmax(output / temperature, dim=0)
        distribution = torch.distributions.Categorical(probabilities)
        
        q = distribution.sample()
        best_act = acts[q]
        best_exp = expect[q]
        Q = output[q]

    if hint: # get first few acts
        qs = torch.argsort(output).flip(dims=(0,))[:5]
        acts_sort = [acts[i] for i in qs]
        print('Candidates:',' | '.join([f'{"pass" if acts_sort[i][0] == "" else acts_sort[i][0]} {str(round(output[qs[i]].item()*100,1)).zfill(5)}%' for i in range(len(qs))]))
    
        exparr = best_exp.detach().numpy().reshape(3,15)[::-1].round(1)
        #print(exparr)

        c0 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in exparr[0]]
        c1 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in exparr[1]]
        c2 = [f"{str(c)}0" if len(str(c)) < 3 else str(c) for c in exparr[2]]

        print(f'{Label[(Turn+1)%3]}:  ','    '.join([str(c) for c in c0]))
        print(f'{Label[(Turn+2)%3]}:  ','    '.join([str(c) for c in c1]))
        print(f'{Label[(Turn+3)%3]}:  ','    '.join([str(c) for c in c2]))

    action = best_act
    return action, Q


def evaluate_action_serial_V2_4_0(Turn, SLM, QV, Initstates, unavail, played_cards,lastmove, Forcemove, history, temperature, Action):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            unavail.unsqueeze(0),
                            played_cards,
                            visible.unsqueeze(0), # new feature
                            torch.full((1, 15), Turn%3),
                            history])
    Bigstate = Bigstate.unsqueeze(1) # model is not changed, so unsqueeze here
    #print(Bigstate)
    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter, lstm_out = SLM(hinput)
    #role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    acts = [Action]
    # generate inputs 2
    model_inter = torch.concat([hinput[0,0:8,0].flatten().unsqueeze(0), # self
                                model_inter, # upper and lower states
                                #role,
                                lstm_out, # lstm encoded history
                                ],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2)[0].flatten()
    Q = output
    return Q
