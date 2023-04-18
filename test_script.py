import torch
import sys
from torch.utils.data import DataLoader
sys.path.append('data/')
sys.path.append('models/')
from my_dataset import MyDataset00000000 as MyDataset
from my_model import MyModel00000000 as MyMultitaskModel

def main():
    # Parameters
    data_path = '/Users/noelkj/Documents/GitHub/AI612-project2/train/testdata/'
    batch_size = 32
    num_epochs = 1
    learning_rate = 1e-3

    # Create the dataset and a data loader
    dataset = MyDataset(data_path=data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = MyMultitaskModel()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Unpack the data and move it to the appropriate device
            inputs = batch["data"].to(model.device)
            targets = batch["label"].to(model.device)

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = model.multitask_loss(targets, outputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss for the current epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

if __name__ == "__main__":
    main()
