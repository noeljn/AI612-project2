import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from . import BaseModel, register_model

@register_model("00000000_model")
class MyModel00000000(BaseModel):

    def __init__(self, input_size=128, hidden_size=512, num_layers=2, dropout_rate=0.2, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the transformer layer
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=hidden_size)

        # Define the batch normalization layer
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        self.task_layers = nn.ModuleDict({
            "short_mortality": nn.Linear(input_size, 1),
            "long_mortality": nn.Linear(input_size, 1),
            "readmission": nn.Linear(input_size, 1),
            "diagnosis_1": nn.Linear(input_size, 1), # 17 diagnoses, each binary
            "diagnosis_2": nn.Linear(input_size, 1),
            "diagnosis_3": nn.Linear(input_size, 1),
            "diagnosis_4": nn.Linear(input_size, 1),
            "diagnosis_5": nn.Linear(input_size, 1),
            "diagnosis_6": nn.Linear(input_size, 1),
            "diagnosis_7": nn.Linear(input_size, 1),
            "diagnosis_8": nn.Linear(input_size, 1),
            "diagnosis_9": nn.Linear(input_size, 1),
            "diagnosis_10": nn.Linear(input_size, 1),
            "diagnosis_11": nn.Linear(input_size, 1),
            "diagnosis_12": nn.Linear(input_size, 1),
            "diagnosis_13": nn.Linear(input_size, 1),
            "diagnosis_14": nn.Linear(input_size, 1),
            "diagnosis_15": nn.Linear(input_size, 1),
            "diagnosis_16": nn.Linear(input_size, 1),
            "diagnosis_17": nn.Linear(input_size, 1),
            "short_los": nn.Linear(input_size, 1),
            "long_los": nn.Linear(input_size, 1),
            "final_acuity": nn.Linear(input_size, 6),
            "imminent_discharge": nn.Linear(input_size, 6),
            "creatinine_level": nn.Linear(input_size, 5),
            "bilirubin_level": nn.Linear(input_size, 5),
            "platelet_level": nn.Linear(input_size, 5),
            "wbc_level": nn.Linear(input_size, 3),
        })
        
    def get_logits(cls, net_output):
        logits = torch.cat([layer for task, layer in net_output.items()], dim=1)
        return logits
    
    def get_targets(self, sample):
        return sample['label']

    def forward(self, data, **kwargs):

        x = data.view(-1, 100, 128)

        # Pass the input through the transformer layer
        x = self.transformer(x)
        
        # Apply mean pooling along the sequence dimension
        x = torch.mean(x, dim=1)

        # Pass the output through the task specific layers
        output = {}
        
        for task, layer in self.task_layers.items():
            output[task] = layer(x)

        return output