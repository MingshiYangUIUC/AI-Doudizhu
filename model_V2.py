import torch
import torch.nn as nn
import torch.nn.functional as F

from base_funcs_V2 import str2state, avail_actions

# network reads in self state, played cards, historical N move, and self action
# output one value (win rate)


class Network_Pcard_V0_0(nn.Module): # this network considers public cards of landlord
                                 # predict opponent states (as frequency of cards) (UPPER, LOWER)
                                 # does not read in action
    def __init__(self, z, y=4, x=15):
        super(Network_Pcard_V0_0, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = 4 # self, unavail, card count, visible, NO action
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=256, batch_first=True)

        # Calculate the size of the non-historical input
        input_size = self.non_hist * y * x + 256  # +256 for the LSTM output
        hidden_size = 512

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
        #print(historical_data.shape)
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


class Network_Qv_Universal_V0_0(nn.Module): # this network uses estimated state to estimate q values of action
                               # use 3 states (SELF, UPPER, LOWER), 1 role, and 1 action (Nelems) (z=5)
                               # cound use more features such as CC
                               # should be simpler

    def __init__(self, z, x=15, hsize=256):
        super(Network_Qv_Universal_V0_0, self).__init__()
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


class Network_Pcard_V1_1(nn.Module): # this network considers public cards of landlord
                                 # predict opponent states (as frequency of cards) (UPPER, LOWER)
                                 # does not read in action
    def __init__(self, z, nh=5, y=4, x=15):
        super(Network_Pcard_V1_1, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = nh # self, unavail, card count, visible, ROLE, NO action
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=256, batch_first=True)

        # Calculate the size of the non-historical input
        input_size = self.non_hist * y * x + 256  # +256 for the LSTM output
        hidden_size = 512

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
    def __init__(self, z, nh=5, y=1, x=15):
        super(Network_Pcard_V1_1, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = nh # self, unavail, card count, visible, ROLE, NO action
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=256, batch_first=True)

        # Calculate the size of the non-historical input
        input_size = self.non_hist * y * x + 256  # +256 for the LSTM output
        hidden_size = 512

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


# wrapper functions for the models defined above


def get_action_serial_V2_1_1(Turn, SLM,QV,Initstates,unavail,lastmove, Forcemove, history, temperature):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            str2state(unavail).unsqueeze(0),
                            CC.unsqueeze(0),
                            visible.unsqueeze(0), # new feature
                            torch.zeros_like(visible).unsqueeze(0) + Turn%3, # role feature
                            history])

    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    # get all actions
    acts = avail_actions(lastmove[0],lastmove[1],Bigstate[0],Forcemove)
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                hinput[:,5].sum(dim=-2),
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

def evaluate_action_serial_V2_1_1(Turn, SLM, QV, Initstates, unavail, lastmove, Forcemove, history, temperature, Action):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            str2state(unavail).unsqueeze(0),
                            CC.unsqueeze(0),
                            visible.unsqueeze(0), # new feature
                            torch.zeros_like(visible).unsqueeze(0) + Turn%3, # role feature
                            history])

    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    acts = [Action]
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                hinput[:,5].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    Q = output
    return Q

def get_action_serial_V2_0_0(Turn, SLM, QV, Initstates,unavail,lastmove, Forcemove, history, temperature):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            str2state(unavail).unsqueeze(0),
                            CC.unsqueeze(0),
                            visible.unsqueeze(0), # new feature
                            history])

    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    # get all actions
    acts = avail_actions(lastmove[0],lastmove[1],Bigstate[0],Forcemove)
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
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

def evaluate_action_serial_V2_0_0(Turn, SLM, QV, Initstates,unavail,lastmove, Forcemove, history, temperature, Action):
    player = Initstates[Turn%3]
    visible = Initstates[-1]

    # get card count
    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.cat([player.unsqueeze(0),
                            str2state(unavail).unsqueeze(0),
                            CC.unsqueeze(0),
                            visible.unsqueeze(0), # new feature
                            history])

    # generate inputs
    hinput = Bigstate.unsqueeze(0)
    model_inter = SLM(hinput)
    role = torch.zeros((model_inter.shape[0],15)) + Turn%3
    
    # get all actions
    acts = [Action]
    # generate inputs 2
    model_inter = torch.concat([hinput[:,0].sum(dim=-2),
                                model_inter,
                                role],dim=-1)
    model_input2 = torch.stack([torch.cat((model_inter.flatten(),str2state(a).sum(dim=0))) for a in acts])

    # get q values
    output = QV(model_input2).flatten()
    #print(output)
    Q = output
    return Q