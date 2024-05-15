import torch
import torch.nn as nn
import torch.nn.functional as F

from base_funcs import str2state, avail_actions

# network reads in self state, played cards, historical N move, and self action
# output one value (win rate)

class Network(nn.Module):
    def __init__(self, z, y=4, x=15, v=1):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        input_size = z * y * x
        hidden_size = 512
        
        self.nhist = z - 4
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, v)
    
    def forward(self, x):
        x = self.flatten(x)  # Flatten the input

        # Compute the first layer
        x = F.relu(self.fc1(x))
        
        # Save output of the first layer for skip connections
        x1 = x
        
        # Process through second and third layers with skip connection after the third layer
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x + x1  # Skip connection from x1 to x3

        # Save output for another skip connection
        x1 = x  # Reuse x1 for new storage to reduce memory usage
        
        # Process through fourth and fifth layers with skip connection after the fifth layer
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = x + x1  # Skip connection from x1 to x5

        # Process sixth layer and final output layer
        x = F.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))
        return x
    

class Network_V2(nn.Module):
    def __init__(self, z, y=4, x=15, v=1):
        super(Network_V2, self).__init__()
        self.y = y
        self.x = x
        self.nhist = z - 4  # Number of historical layers to process with LSTM
        lstm_input_size = y * x  # Assuming each layer is treated as one sequence element

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=256, batch_first=True)

        # Calculate the size of the non-historical input
        non_historical_layers = 4
        input_size = non_historical_layers * y * x + 256  # +256 for the LSTM output
        hidden_size = 512

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, v)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Extract the historical layers for LSTM processing
        historical_data = x[:, 3:3+self.nhist, :, :]  # Historical layers are from index 3 to 17
        historical_data = historical_data.flip(dims=[1])
        historical_data = historical_data.reshape(-1, self.nhist, self.y * self.x)  # Reshape for LSTM

        # Process historical layers with LSTM
        lstm_out, _ = self.lstm(historical_data)
        lstm_out = lstm_out[:, -1, :]  # Use only the last output of the LSTM

        # Extract and flatten the non-historical part of the input
        non_historical_data = torch.cat((x[:, :3, :, :], x[:, 3+self.nhist:, :, :]), dim=1)  # Keeping layers outside of 3 to 17
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
        x = torch.sigmoid(self.fc7(x))
        return x
    
class Network_V3(nn.Module): # this network considers public cards of landlord
    def __init__(self, z, y=4, x=15, v=1):
        super(Network_V3, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = 5
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
        self.fc7 = nn.Linear(hidden_size, v)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Extract the historical layers for LSTM processing
        historical_data = x[:, self.non_hist-1:self.non_hist-1+self.nhist, :, :]  # Historical layers are from index 3 to 17
        historical_data = historical_data.flip(dims=[1])
        historical_data = historical_data.reshape(-1, self.nhist, self.y * self.x)  # Reshape for LSTM

        # Process historical layers with LSTM
        lstm_out, _ = self.lstm(historical_data)
        lstm_out = lstm_out[:, -1, :]  # Use only the last output of the LSTM

        # Extract and flatten the non-historical part of the input
        non_historical_data = torch.cat((x[:, :self.non_hist-1, :, :], x[:, self.non_hist-1+self.nhist:, :, :]), dim=1)  # Keeping layers outside of 3 to 17
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
        x = torch.sigmoid(self.fc7(x))
        return x


class Network_V3_Unified(nn.Module):
    def __init__(self, z, y=4, x=15, v=1, num_players=3):
        super(Network_V3_Unified, self).__init__()
        self.y = y
        self.x = x
        self.non_hist = 5
        self.nhist = z - self.non_hist  # Number of historical layers to process with LSTM
        lstm_input_size = y * x

        # LSTM to process the historical data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=256, batch_first=True)

        # Adding player identifier vector size (num_players)
        input_size = self.non_hist * y * x + 256 + num_players  # +256 for the LSTM output, +num_players for player identifier

        # Define the sizes of layers in the network
        hidden_size = 512

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, v)
        self.flatten = nn.Flatten()

    def forward(self, x, player_id):
        # Process historical data
        historical_data = x[:, self.non_hist-1:self.non_hist-1+self.nhist, :, :]
        historical_data = historical_data.flip(dims=[1])
        historical_data = historical_data.reshape(-1, self.nhist, self.y * self.x)
        lstm_out, _ = self.lstm(historical_data)
        lstm_out = lstm_out[:, -1, :]  # Use only the last output of the LSTM

        # Process non-historical data
        non_historical_data = torch.cat((x[:, :self.non_hist-1, :, :], x[:, self.non_hist-1+self.nhist:, :, :]), dim=1)
        non_historical_data = non_historical_data.reshape(-1, non_historical_data.shape[1] * self.y * self.x)

        # Concatenate LSTM output, non-historical data, and player identifier
        player_id = player_id.reshape(-1, player_id.shape[1])  # Ensure player_id is correctly shaped
        #print(lstm_out.shape,non_historical_data.shape,player_id.shape)
        x = torch.cat((lstm_out, non_historical_data, player_id), dim=1)

        # Feed through the fully connected layers
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
        x = torch.sigmoid(self.fc7(x))
        return x


# wrapper for network_V2

def get_action_serial_V1_0_0(Turn, model, Initstates,unavail,lastmove, Forcemove, history, temperature):
    player = Initstates[Turn%3]

    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.concat([player.unsqueeze(0),str2state(unavail).unsqueeze(0),CC.unsqueeze(0),history])

    acts = avail_actions(lastmove[0],lastmove[1],Bigstate[0],Forcemove)

    hinput = torch.concat([torch.concat([Bigstate,str2state(a[0]).unsqueeze(0)]).unsqueeze(0) for a in acts])

    output = model(hinput).flatten()

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


def get_action_serial_V1_1_0(Turn, model, Initstates,unavail,lastmove, Forcemove, history, temperature):
    player = Initstates[Turn%3]

    visible = Initstates[-1]

    card_count = [int(p.sum()) for p in Initstates]
    #print(card_count)
    CC = torch.zeros((4,15))
    CC[0][:min(card_count[0],15)] = 1
    CC[1][:min(card_count[1],15)] = 1
    CC[2][:min(card_count[2],15)] = 1
    #print(CC)

    # get action
    Bigstate = torch.concat([player.unsqueeze(0),str2state(unavail).unsqueeze(0),CC.unsqueeze(0),visible.unsqueeze(0),history])

    acts = avail_actions(lastmove[0],lastmove[1],Bigstate[0],Forcemove)

    hinput = torch.concat([torch.concat([Bigstate,str2state(a[0]).unsqueeze(0)]).unsqueeze(0) for a in acts])

    output = model(hinput).flatten()

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