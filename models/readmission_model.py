from typing import Dict
import torch
import torch.nn as nn
from . import BaseModel, register_model

@register_model("00000000_readmission_model")
class MyReadmissionModel00000000(BaseModel):

    def __init__(self, input_size=12800, hidden_size=512, num_classes=2, **kwargs):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Task-specific output layer
        self.readmission = nn.Linear(hidden_size, num_classes)
        
    def get_logits(self, net_output):
        return net_output
    
    def get_targets(self, sample):
        return sample["labels"]
    
    def forward(self, data_key, **kwargs):
        x = self.relu(self.fc1(data_key))
        readmission_out = self.readmission(x)
        
        # Return the logits as a dictionary
        logits = {
            "readmission": readmission_out,
        }
        
        return logits
