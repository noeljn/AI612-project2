from . import register_dataset, BaseDataset
#from dataset_registry import register_dataset
#from dataset import BaseDataset
import os
import torch
from typing import List, Dict


@register_dataset("00000000_dataset")
class MyDataset00000000(BaseDataset):
    def __init__(
        self,
        data_path: str, # data_path should be a path to the processed features
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading data from {data_path} to {device}")
        super().__init__()
        self.data = torch.load(os.path.join(data_path, 'features.pkl'), map_location=device)
        self.labels = torch.load(os.path.join(data_path, 'labels.pkl'), map_location=device)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data = self.data[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return {"data": data, "label": label}

    def __len__(self) -> int:
        return len(self.data)

    def collator(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        data = torch.stack([sample['data'] for sample in samples], dim=0)
        labels = torch.stack([sample['label'] for sample in samples], dim=0)

        return {"data": data, "label": labels}
