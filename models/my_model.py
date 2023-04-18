from typing import Dict
import torch
import torch.nn as nn
#from . import BaseModel, register_model
from models import BaseModel
import torch.nn.functional as F


#@register_model("00000000_model")
class MyModel00000000(BaseModel):

    def __init__(self, input_size=12800, hidden_size=512, num_layers=1, dropout_rate=0.2, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the batch normalization layer
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        self.task_layers = nn.ModuleDict({
            "short_mortality": nn.Linear(hidden_size, 2),
            "long_mortality": nn.Linear(hidden_size, 2),
            "readmission": nn.Linear(hidden_size, 2),
            "diagnosis": nn.Linear(hidden_size, 34), # 17 diagnoses, each binary
            "short_los": nn.Linear(hidden_size, 2),
            "long_los": nn.Linear(hidden_size, 2),
            "final_acuity": nn.Linear(hidden_size, 6),
            "imminent_discharge": nn.Linear(hidden_size, 6),
            "creatinine_level": nn.Linear(hidden_size, 5),
            "bilirubin_level": nn.Linear(hidden_size, 5),
            "platelet_level": nn.Linear(hidden_size, 5),
            "wbc_level": nn.Linear(hidden_size, 3),
        })
        
    def get_logits(cls, net_output):
        logits = []
        idx = 0
        for task_size in cls.task_sizes:
            logits.append(net_output[:, idx:idx + task_size])
            idx += task_size
        return logits
    
    def get_targets(self, sample):
        return sample["labels"]
    
    def forward(self, x, **kwargs):

        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass the input through the GRU layer 
        x, _ = self.gru(x, h0)
        
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
        
    def multitask_loss(self, predictions, targets):
        # Define loss functions for each task
        loss_functions = {
            "short_mortality": F.cross_entropy,
            "long_mortality": F.cross_entropy,
            "readmission": F.cross_entropy,
            "diagnosis": F.binary_cross_entropy_with_logits,
            "short_los": F.cross_entropy,
            "long_los": F.cross_entropy,
            "final_acuity": F.cross_entropy,
            "imminent_discharge": F.cross_entropy,
            "creatinine_level": F.cross_entropy,
            "bilirubin_level": F.cross_entropy,
            "platelet_level": F.cross_entropy,
            "wbc_level": F.cross_entropy,
        }
        
        # Calculate loss for each task
        losses = {task: loss_functions[task](predictions[task], targets[task]) for task in predictions.keys()}
        
        # Combine the losses
        total_loss = sum(losses.values())
        
        return total_loss
