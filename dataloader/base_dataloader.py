from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None, target_transform=None) -> None:
        self.data = data_dir
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return the number of samples in the custom dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Loda and return a data sample from the custom dataset at given index."""
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label