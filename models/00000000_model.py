from typing import Dict
import torch
import torch.nn as nn
from . import BaseModel, register_model

@register_model("00000000_model")
class MyModel00000000(BaseModel):

    def __init__(self, input_size=12800, hidden_size=512, **kwargs):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # Task-specific output layers
        self.short_mortality = nn.Linear(hidden_size, 2)
        self.long_mortality = nn.Linear(hidden_size, 2)
        self.readmission = nn.Linear(hidden_size, 2)
        self.diagnosis = nn.Linear(hidden_size, 17)
        self.short_los = nn.Linear(hidden_size, 2)
        self.long_los = nn.Linear(hidden_size, 2)
        self.final_acuity = nn.Linear(hidden_size, 6)
        self.imminent_discharge = nn.Linear(hidden_size, 6)
        self.creatinine_level = nn.Linear(hidden_size, 5)
        self.bilirubin_level = nn.Linear(hidden_size, 5)
        self.platelet_level = nn.Linear(hidden_size, 5)
        self.wbc_level = nn.Linear(hidden_size, 3)

        self.sigmoid = nn.Sigmoid()
        
    def get_logits(cls, net_output):
        logits = []
        for key in sorted(net_output.keys()):
            logits.append(net_output[key])
        return torch.cat(logits, dim=1)
    
    def get_targets(self, sample):
        return sample["labels"]
    
    def forward(self, data_key, **kwargs):
        x = self.relu(self.fc1(data_key))

        short_mortality_out = self.short_mortality(x)
        long_mortality_out = self.long_mortality(x)
        readmission_out = self.readmission(x)
        diagnosis_out = self.sigmoid(self.diagnosis(x))
        short_los_out = self.short_los(x)
        long_los_out = self.long_los(x)
        final_acuity_out = self.final_acuity(x)
        imminent_discharge_out = self.imminent_discharge(x)
        creatinine_level_out = self.creatinine_level(x)
        bilirubin_level_out = self.bilirubin_level(x)
        platelet_level_out = self.platelet_level(x)
        wbc_level_out = self.wbc_level(x)
        
         # Combine the logits into a single dictionary
        logits = {
            "short_mortality": short_mortality_out,
            "long_mortality": long_mortality_out,
            "readmission": readmission_out,
            "diagnosis": diagnosis_out,
            "short_los": short_los_out,
            "long_los": long_los_out,
            "final_acuity": final_acuity_out,
            "imminent_discharge": imminent_discharge_out,
            "creatinine_level": creatinine_level_out,
            "bilirubin_level": bilirubin_level_out,
            "platelet_level": platelet_level_out,
            "wbc_level": wbc_level_out,
        }
        
        return logits