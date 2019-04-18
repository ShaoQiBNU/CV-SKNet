################# load packages #################
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, batch_size):

        self.batch_size = batch_size

        ######### set data transform ###########
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

        ######### download data ###########
        self.data_train = datasets.MNIST(root="MNIST/", transform=self.transform, train=True, download=True)
        self.data_test = datasets.MNIST(root="MNIST/", transform=self.transform, train=False)

    ################# preprocess data #################
    def load_train_data(self):

        ######### data loader ###########
        data_loader_train = torch.utils.data.DataLoader(dataset=self.data_train, batch_size=self.batch_size, shuffle=True)

        return data_loader_train

    ################# preprocess data #################
    def load_test_data(self):

        ######### data loader ###########
        data_loader_test = torch.utils.data.DataLoader(dataset=self.data_test, batch_size=self.batch_size)

        return data_loader_test