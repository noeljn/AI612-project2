from typing import Dict
import torch
import torch.nn as nn
from . import BaseModel, register_model
#from models import BaseModel
import torch.nn.functional as F


@register_model("00000000_model")
class MyModel00000000(BaseModel):

    def __init__(self, input_size=128, hidden_size=512, num_layers=1, dropout_rate=0.2, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the lstm layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the batch normalization layer
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        self.task_layers = nn.ModuleDict({
            "short_mortality": nn.Linear(hidden_size, 1),
            "long_mortality": nn.Linear(hidden_size, 1),
            "readmission": nn.Linear(hidden_size, 1)
        })
        for i in range(1, 18):
            self.task_layers[f"diagnosis_{i}"] = nn.Linear(hidden_size, 1)
        self.task_layers.update({
            "short_los": nn.Linear(hidden_size, 1),
            "long_los": nn.Linear(hidden_size, 1),
            "final_acuity": nn.Linear(hidden_size, 6),
            "imminent_discharge": nn.Linear(hidden_size, 6),
            "creatinine_level": nn.Linear(hidden_size, 5),
            "bilirubin_level": nn.Linear(hidden_size, 5),
            "platelet_level": nn.Linear(hidden_size, 5),
            "wbc_level": nn.Linear(hidden_size, 3),
        })
        

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()
        
        # Softmax activation function
        self.softmax = nn.Softmax(dim=1)
        
    def get_logits(self, net_output):
        # Output (batch, 52)
        # for task in net_output.keys():
        #         if net_output[task].shape[1] == 1:
        #             net_output[task] = self.sigmoid(net_output[task])
        #         else:
        #             net_output[task] = self.softmax(net_output[task])
        # print the shape o each task
        # for task in net_output.keys():
        #     print(task, net_output[task].shape)
        return torch.cat([net_output[task] for task in net_output.keys()], dim=1)
    
    def get_targets(self, sample):
        return sample["label"]
    
    def forward(self, **kwargs):
        x = kwargs["data"]

        x = x.view(-1, 100, 128)

        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass the input through the LSTM layer
        x, _ = self.lstm(x, (h0, c0))
        
        # Apply batch normalization
        x = self.batchnorm(x[:, -1, :])

        # Apply ReLU activation function
        x = self.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass the output through the task specific layers
        output = {}
        for task, layer in self.task_layers.items():
            output[task] = layer(x)

        return output
        
    def multitask_loss(self, cls, predictions, targets):
        # Define loss functions for each task
        loss_functions = {
            "short_mortality": nn.CrossEntropyLoss(),
            "long_mortality": nn.CrossEntropyLoss(),
            "readmission": nn.CrossEntropyLoss(),
            "diagnosis": nn.BCEWithLogitsLoss(),
            "short_los": nn.CrossEntropyLoss(),
            "long_los": nn.CrossEntropyLoss(),
            "final_acuity": nn.CrossEntropyLoss(),
            "imminent_discharge": nn.CrossEntropyLoss(),
            "creatinine_level": nn.CrossEntropyLoss(),
            "bilirubin_level": nn.CrossEntropyLoss(),
            "platelet_level": nn.CrossEntropyLoss(),
            "wbc_level": nn.CrossEntropyLoss(),
        }
        
        # Calculate loss for each task
        losses = {}
        idx = 0
        for task, prediction in predictions.items():
            i = cls[task]
            target_task = [t[idx:i] for t in targets]
            # Reshape the target
            target_task = torch.stack(target_task)
            losses[task] = loss_functions[task](prediction, target_task)
            idx = i
        
        # Combine the losses
        total_loss = sum(losses.values())
        
        return total_loss