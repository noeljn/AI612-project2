import os
import torch
from . import BaseDataset, register_dataset
from typing import List, Dict

@register_dataset("00000000_dataset")
class MyDataset00000000(BaseDataset):
    def __init__(
        self,
        data_path: str, # data_path should be a path to the processed features
        **kwargs,
    ):
        
        print(f"Loading data from {data_path}")
        
        super().__init__()   
        
        self.data = torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'features.pkl'))
        self.labels = torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'labels.pkl'))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return {"data": data, "label": label}

    def __len__(self) -> int:
        return len(self.data)

    def collator(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        data = torch.stack([sample['data'] for sample in samples], dim=0)
        labels = torch.stack([sample['label'] for sample in samples], dim=0)

        return {"data": data, "label": labels}
