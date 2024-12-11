import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class CustomNet(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 100, output_size = 2):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        x = self.fc2(x)
        x = self.relu(self.bn2(x))
        x = self.fc3(x)
        x = self.relu(self.bn3(x))
        x = self.output(x)
        return x

class SolarNet(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 100, output_size = 2):
        super(SolarNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        x = self.fc2(x)
        x = self.relu(self.bn2(x))
        x = self.fc3(x)
        x = self.relu(self.bn3(x))
        
        with torch.no_grad():
            self.output.weight.data.clamp_(min=0)
            if self.output.bias is not None:
                self.output.bias.data.clamp_(min=0)
        
        x = self.output(x)
        return x

    
class MVENet(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 100, output_size = 2):
        super(MVENet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        x = self.fc2(x)
        x = self.relu(self.bn2(x))
        x = self.fc3(x)
        x = self.relu(self.bn3(x))
        x = self.output(x)
        mean = x[:,0]
        variance = x[:,1]        
        return mean, variance
    
    
class SolarkstepaheadNet(nn.Module):
    def __init__(self, input_window_size = 24, hidden_size = 100, predicted_step = 1):
        super(SolarkstepaheadNet, self).__init__()
        
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.input_window_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 2*self.predicted_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        x = self.fc2(x)
        x = self.relu(self.bn2(x))
        x = self.fc3(x)
        x = self.relu(self.bn3(x))
        x = self.output(x)
        return x

class SolarkstepaheadNet_exoinput(nn.Module):
    def __init__(self, lag_input_window_size=24, exo_input_window_size=12, hidden_size=100, predicted_step=1):
        super(SolarkstepaheadNet_exoinput, self).__init__()
        
        self.lag_input_window_size = lag_input_window_size
        self.exo_input_window_size = exo_input_window_size
        self.hidden_size = hidden_size
        self.predicted_step = predicted_step
        self.num_exo_input = self.exo_input_window_size // self.predicted_step
        
        # Common layers for lag inputs
        self.fc_common1 = nn.Linear(self.lag_input_window_size, self.hidden_size)
        self.bn_common1 = nn.BatchNorm1d(self.hidden_size)
        self.fc_common2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn_common2 = nn.BatchNorm1d(self.hidden_size)
        
        # Exogenous layers for each step
        self.exo_layers1 = nn.ModuleList([
            nn.Linear(self.hidden_size + self.num_exo_input, self.hidden_size) for _ in range(predicted_step)
        ])
        self.bn_exo_layers1 = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size) for _ in range(predicted_step)
        ])
        
        self.exo_layers2 = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) for _ in range(predicted_step)
        ])
        self.bn_exo_layers2 = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size) for _ in range(predicted_step)
        ])
        
        # Output layers for each step (2 outputs per step for prediction intervals)
        self.output_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, 2) for _ in range(predicted_step)
        ])
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        common_input = x[:,:self.lag_input_window_size]
        exo_input = x[:,self.lag_input_window_size:]
        
        batch_size = common_input.size(0)
        
        # Exogenous inputs for each step ahead
        exo_ahead = [exo_input[:, i*self.num_exo_input:(i+1)*self.num_exo_input] for i in range(self.predicted_step)]
        
        # Process common input with ReLU and BatchNorm after each layer
        common_input = self.relu(self.bn_common1(self.fc_common1(common_input)))
        common_input = self.relu(self.bn_common2(self.fc_common2(common_input)))
                
        outputs = []
        
        # Loop through each step ahead
        for i in range(self.predicted_step):
            # Create input for this step (concatenating common_input[i] and corresponding exo input)
            step_input = torch.cat((common_input, exo_ahead[i]), dim=1)
            
            # Process through exogenous layer for this step with ReLU and BatchNorm
            step_hidden = self.relu(self.bn_exo_layers1[i](self.exo_layers1[i](step_input)))
            step_hidden = self.relu(self.bn_exo_layers2[i](self.exo_layers2[i](step_hidden)))
            
            # Get prediction interval for this step with ReLU and BatchNorm
            step_output = self.output_layers[i](step_hidden)
            outputs.append(step_output)
        
        # Concatenate the results from all steps into shape (N, 2 * predicted_step)
        final_output = torch.cat(outputs, dim=1)
        return final_output
    
class SolarkstepaheadNet_LSTM_exoinput(nn.Module):
    def __init__(self, lag_input_window_size = 24, exo_input_window_size = 12, num_lag_features= 1, num_layers = 1,
                 hidden_size = 100, lstm_hidden_size = 100, predicted_step = 1):
        super(SolarkstepaheadNet_LSTM_exoinput, self).__init__()
        
        self.lag_input_window_size = lag_input_window_size
        self.exo_input_window_size = exo_input_window_size
        self.hidden_size = hidden_size
        self.predicted_step = predicted_step
        self.num_exo_input = self.exo_input_window_size // self.predicted_step
        self.num_lag_features = num_lag_features
        self.num_layers = num_layers
        
        # Common layers for lag inputs
        self.fc_common1 = nn.Linear(self.lag_input_window_size, self.hidden_size)
        self.bn_common1 = nn.BatchNorm1d(self.hidden_size)
        self.fc_common2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn_common2 = nn.BatchNorm1d(self.hidden_size)
        
        # Define the LSTM layer
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=self.num_lag_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        
        # Batch Normalization for the LSTM output
        self.bn_lstm = nn.BatchNorm1d(self.lstm_hidden_size)
        
        # Exogenous layers for each step
        self.exo_layers1 = nn.ModuleList([
            nn.Linear(self.lstm_hidden_size + self.num_exo_input, self.hidden_size) for _ in range(predicted_step)
        ])
        self.bn_exo_layers1 = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size) for _ in range(predicted_step)
        ])
        
        self.exo_layers2 = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) for _ in range(predicted_step)
        ])
        self.bn_exo_layers2 = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size) for _ in range(predicted_step)
        ])
        
        # Output layers for each step (2 outputs per step for prediction intervals)
        self.output_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, 2) for _ in range(predicted_step)
        ])
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        common_input = x[:, :self.lag_input_window_size]
        exo_input = x[:, self.lag_input_window_size:]
        batch_size = common_input.size(0)
        
        # Reshape common input to [batch_size, num_lag_features, num_each_feature_lag]
        num_each_feature_lag = self.lag_input_window_size // self.num_lag_features
        common_input = common_input.view(batch_size, self.num_lag_features, num_each_feature_lag).transpose(1, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(common_input)  # lstm_out has shape (batch_size, input_window_size (4) , hidden_size)
        # Use the output of the last time step (sequence length = input_window_size) for prediction
        common_input = lstm_out[:, -1, :]  # lstm_out_last has shape (batch_size, hidden_size)
        # Apply Batch Normalization after LSTM
        common_input = self.relu(self.bn_lstm(common_input))
        
        # Exogenous inputs for each step ahead
        exo_ahead = [exo_input[:, i * self.num_exo_input:(i + 1) * self.num_exo_input] for i in range(self.predicted_step)]
                
        outputs = []
        
        # Loop through each step ahead
        for i in range(self.predicted_step):
            # Create input for this step (concatenating common_input and corresponding exo input)
            step_input = torch.cat((common_input, exo_ahead[i]), dim=1)
            
            # Process through exogenous layer for this step with ReLU and BatchNorm
            step_hidden = self.relu(self.bn_exo_layers1[i](self.exo_layers1[i](step_input)))
            step_hidden = self.relu(self.bn_exo_layers2[i](self.exo_layers2[i](step_hidden)))
            
            # Get prediction interval for this step
            step_output = self.output_layers[i](step_hidden)
            outputs.append(step_output)
        
        # Concatenate the results from all steps into shape (N, 2 * predicted_step)
        final_output = torch.cat(outputs, dim=1)
        return final_output