import torch
import torchvision
import torchvision.transforms as transforms

# from: https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/3
class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def split_train_test(path_to_data='./dataset/plantvillage/', 
                     test_split=0.1, 
                     train_transforms=transforms.ToTensor(),
                     test_transforms=transforms.ToTensor()):

    '''
    Split train and test partitions given a folder in ImageFolder dataset format

    :param path_to_data (string): string with path to image folder
    :param test_split (float): percentage (0-1.0) of data to test fold
    :param train_transforms (torchvision.transforms.Compose): transforms operations to be used in train data
    :param val_transforms (torchvision.transforms.Compose): transforms operations to be used in test data
    '''

    dataset = torchvision.datasets.ImageFolder(root=path_to_data)
    print(len(dataset))
    # lengths = [int((1 - test_split) * len(dataset)), int(test_split * len(dataset))]
    lengths = [48875, 5430]
    train_subset, test_subset = torch.utils.data.random_split(dataset, lengths)

    train_dataset = DatasetFromSubset(subset=train_subset, transform=train_transforms)
    test_dataset = DatasetFromSubset(subset=test_subset, transform=test_transforms)

    return train_dataset, test_dataset

def get_dataloader(dataset=None, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    '''
    Function to construct a dataloader given a dataset

    :param dataset (torch.utils.data.Dataset): dataset used
    :param batch_size (int): batch size
    :param shuffle (bool): if you'd like to shuffle the dataset
    :param num_workers (int): number of threads in cpu
    :param pin_memory (bool): load your images in gpu
    '''

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    

    

