import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import ImageEnhance
from sklearn import preprocessing
from torchvision.transforms import transforms
from sklearn.utils import shuffle

Firstfeature = transforms.Compose([
        transforms.ColorJitter(hue=0.3),
        torchvision.transforms.ToTensor()])

Secondfeature = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=0.3),
        torchvision.transforms.ToTensor()])
def loadcifar():


    cifar10_original = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        #transform=transforms.ToTensor()
    )

    cifar10_color = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        #transform=Firstfeature
    )

    cifar10_trans = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        #transform=Secondfeature
    )


    x_S1 = torch.Tensor(cifar10_original.data)
    x_S3 = torch.Tensor(cifar10_trans.data)
    x_S2 = torch.Tensor(cifar10_color.data)
    x_S1 = torch.transpose(x_S1, 3, 2)
    x_S1 = torch.transpose(x_S1, 2, 1)
    x_S2 = torch.transpose(x_S2, 3, 2)
    x_S2 = torch.transpose(x_S2, 2, 1)
    x_S3 = torch.transpose(x_S3, 3, 2)
    x_S3 = torch.transpose(x_S3, 2, 1)

    x_S2 = transforms.ColorJitter(hue=0.3)(x_S2)
    x_S3 = transforms.ColorJitter(hue=0.3)(x_S3)
    x_S3 = transforms.RandomHorizontalFlip(p=0.5)(x_S3)

    y_S1, y_S2,y_S3 = torch.Tensor(cifar10_original.targets), torch.Tensor(cifar10_color.targets), torch.Tensor(cifar10_trans.targets)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    x_S3, y_S3 = shuffle(x_S3, y_S3, random_state=30)

    return x_S1, y_S1, x_S2, y_S2,x_S3,y_S3

def loadsvhn():
    svhn_original =  torchvision.datasets.SVHN('./data', split='train', download=True,
                               transform=transforms.Compose([ transforms.ToTensor()]))
    svhn_color = torchvision.datasets.SVHN(
        root='./data',
        split="train",
        download=True,
        transform=Firstfeature
    )
    svhn_trans = torchvision.datasets.SVHN(
        root='./data',
        split="train",
        download=True,
        transform=Secondfeature
    )
    x_S1 = torch.Tensor(svhn_original.data)
    x_S2 = torch.Tensor(svhn_color.data)
    x_S3 = torch.Tensor(svhn_trans.data)
    x_S2 = transforms.ColorJitter(hue=0.3)(x_S2)
    x_S3 = transforms.ColorJitter(hue=0.3)(x_S3)
    x_S3 = transforms.RandomHorizontalFlip(p=0.5)(x_S3)
    y_S1, y_S2, y_S3 = svhn_original.labels, svhn_color.labels,svhn_trans.labels

    y_S1, y_S2, y_S3 = torch.Tensor(y_S1), torch.Tensor(y_S1), torch.Tensor(y_S3)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    x_S3, y_S3 = shuffle(x_S3, y_S3, random_state=30)
    return x_S1, y_S1, x_S2, y_S2,x_S3,y_S3

def loadmagic():
    data = pd.read_csv(r"./data/magic04_X.csv", header=None).values
    label = pd.read_csv(r"./data/magic04_y.csv", header=None).values
    for i in label:
        if i[0] == -1:
            i[0] = 0
    rd1 = np.random.RandomState(1314)
    data = preprocessing.scale(data)
    matrix1 = rd1.random((10, 30))
    matrix2 = rd1.random((30, 50))

    x_S2 = np.dot(data, matrix1)
    x_S3 = np.dot(x_S2, matrix2)
    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    x_S3 = torch.sigmoid(torch.Tensor(x_S3))

    y_S1, y_S2, y_S3 = torch.Tensor(label), torch.Tensor(label),torch.Tensor(label)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=50)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=50)
    x_S3, y_S3 = shuffle(x_S3, y_S3, random_state=50)
    return x_S1, y_S1, x_S2, y_S2, x_S3, y_S3

def loadadult():
    df1 = pd.read_csv(r"./data/adult.data", header=1)
    df1.columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    le = preprocessing.LabelEncoder()
    le.fit(df1.o)
    df1['o'] = le.transform(df1.o)
    le.fit(df1.b)
    df1['b'] = le.transform(df1.b)
    le.fit(df1.d)
    df1['d'] = le.transform(df1.d)
    le.fit(df1.f)
    df1['f'] = le.transform(df1.f)
    le.fit(df1.g)
    df1['g'] = le.transform(df1.g)
    le.fit(df1.h)
    df1['h'] = le.transform(df1.h)
    le.fit(df1.i)
    df1['i'] = le.transform(df1.i)
    le.fit(df1.j)
    df1['j'] = le.transform(df1.j)
    le.fit(df1.n)
    df1['n'] = le.transform(df1.n)
    data = np.array(df1.iloc[:, :-1])
    label = np.array(df1.o)
    rd1 = np.random.RandomState(1314)
    data = preprocessing.scale(data)
    matrix1 = rd1.random((14, 30))
    matrix2 = rd1.random((30, 40))

    x_S2 = np.dot(data, matrix1)
    x_S3 = np.dot(x_S2, matrix2)
    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    x_S3 = torch.sigmoid(torch.Tensor(x_S3))

    y_S1, y_S2, y_S3 = torch.Tensor(label), torch.Tensor(label), torch.Tensor(label)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=50)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=50)
    x_S3, y_S3 = shuffle(x_S3, y_S3, random_state=50)
    return x_S1, y_S1, x_S2, y_S2, x_S3, y_S3


def loadreuter(name):
    x_S1 = torch.Tensor(torch.load('D:/old3s_original/Reuter/data/' + name +'/x_S1_NonLinear'))
    y_S1 = torch.Tensor(torch.load('D:/old3s_original/Reuter/data/' + name +'/y_S1_multiLinear'))
    x_S2 = torch.Tensor(torch.load('D:/old3s_original/Reuter/data/' + name +'/x_S2_NonLinear'))
    y_S2 = torch.Tensor(torch.load('D:/old3s_original/Reuter/data/' + name +'/y_S2_multiLinear'))
    x_S3 = torch.Tensor(torch.load('D:/old3s_original/Reuter/data/' + 'FR_IT' + '/x_S2_NonLinear'))
    y_S3 = torch.Tensor(torch.load('D:/old3s_original/Reuter/data/' + 'FR_IT' + '/y_S2_multiLinear'))
    return x_S1, y_S1, x_S2, y_S2, x_S3, y_S3

def loadmnist():
    mnist_original = torchvision.datasets.FashionMNIST(
        root='./data',
        download=True,
        train=True,
        # Simply put the size you want in Resize (can be tuple for height, width)
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        ),
    )
    mnist_color = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [

                transforms.ColorJitter(hue=0.3),
                torchvision.transforms.ToTensor()]
        ),
    )
    mnist_trans = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                transforms.ColorJitter(hue=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor()]
        ),
    )
    x_S1 = mnist_original.data.reshape(60000,1,28,28)
    x_S2 = mnist_color.data.reshape(60000,1,28,28)
    x_S3 = mnist_trans.data.reshape(60000,1,28,28)

    x_S2 = transforms.ColorJitter(hue=0.3)(x_S2)

    x_S3 = transforms.ColorJitter(hue=0.3)(x_S3)
    x_S3 = transforms.RandomHorizontalFlip(p=0.5)(x_S3)

    y_S1, y_S2, y_S3 = mnist_original.targets, mnist_color.targets, mnist_trans.targets
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=1000)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=1000)
    x_S3, y_S3 = shuffle(x_S3, y_S3, random_state=1000)
    return x_S1, y_S1, x_S2, y_S2, x_S3, y_S3




