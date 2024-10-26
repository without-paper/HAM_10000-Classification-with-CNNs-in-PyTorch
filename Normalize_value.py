import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: Custom class Dataset (or ImageFolder will do)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'HAM10000_images/train', transform=transforms.ToTensor())
    print(getStat(train_dataset))

