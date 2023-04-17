import os
import torch
from torch.utils.data import DataLoader, random_split
from my_dataset import MyDataset00000000

def main():
    # Set the path to your preprocessed data
    data_path = '/Users/noelkj/Documents/GitHub/AI612-project2/train/testdata/'

    MyDataset = MyDataset00000000

    # Create an instance of your custom dataset class
    dataset = MyDataset(data_path=data_path)

    # Optionally, split the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for the train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=dataset.collator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=dataset.collator)

    # Iterate over the train DataLoader (simulate one training epoch)
    for batch_idx, batch in enumerate(train_loader):
        data = batch['data_key']
        labels = batch['label']
        print(f"Batch {batch_idx + 1}: data shape = {data.shape}, labels shape = {labels.shape}")

    # Iterate over the validation DataLoader (simulate one validation epoch)
    for batch_idx, batch in enumerate(val_loader):
        data = batch['data_key']
        labels = batch['label']
        print(f"Batch {batch_idx + 1}: data shape = {data.shape}, labels shape = {labels.shape}")

if __name__ == "__main__":
    main()
