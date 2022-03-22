import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model.cnn import CNN

from config.config_loader import load_config
config = load_config()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_model(nn):
    model = nn
    model.to(torch.device(device))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    return model, criterion, optimizer


def train(loader, model, criterion, optimizer):
    model.train()
    for idx, data in enumerate(loader):
        optimizer.zero_grad()    # Clear gradients

        x = data[0].to(torch.device(device))
        y = data[1].to(torch.device(device))
        
        logits = model(x)             # Feedforward
        loss = criterion(logits, y)   # Compute gradients

        loss.backward()          # Backward pass
        optimizer.step()         # Update model parametersss


@torch.no_grad()
def validation(loader, model, criterion):
    model.eval()
    for idx, data in enumerate(loader):
        x = data[0].to(torch.device(device))
        y = data[1].to(torch.device(device))

        logits = model(x)
        loss = criterion(logits, y)  


if __name__ == "__main__": 
    EPOCH = config['epoch']

    # Image Dataset
    training_data = datasets.MNIST(
        root='../dataset/',
        train=True,
        download=True,
        transform=ToTensor())

    testing_data = datasets.MNIST(
        root='../dataset/',
        train=False,
        download=True,
        transform=ToTensor())

    train_loader = DataLoader(training_data, batch_size=3, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=3, shuffle=True)

    model, criterion, optimizer = build_model(CNN(in_channels=1, out_channels=10))

    for epoch in range(1, EPOCH+1):
        # Training & Validation
        train(train_loader, model, criterion, optimizer)
        validation(test_loader, model, criterion)