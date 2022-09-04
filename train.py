import random
import time
import torch
import numpy as np
import argparse
from model import *
from loaddatasets import *
from model_vae import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-DataName', action='store', dest='DataName', default='adult')
    parser.add_argument('-AutoEncoder', action='store', dest='AutoEncoder', default='AE')
    parser.add_argument('-beta', action='store', dest='beta', default=0.9)
    parser.add_argument('-eta', action='store', dest='eta', default=-0.01)
    parser.add_argument('-learningrate', action='store', dest='learningrate', default=0.01)
    parser.add_argument('-RecLossFunc', action='store', dest='RecLossFunc', default='BCE')
    args = parser.parse_args()
    learner = OLD3S(args)
    learner.train()


class OLD3S:
    def __init__(self, args):
        '''
            Data is stored as list of dictionaries.
            Label is stored as list of scalars.
        '''
        self.datasetname = args.DataName
        self.autoencoder = args.AutoEncoder
        self.beta = args.beta
        self.eta = args.eta
        self.learningrate = args.learningrate
        self.RecLossFunc = args.RecLossFunc

    def train(self):
        if self.datasetname == 'cifar':
            t = time.time()

            print('cifar trainning starts')
            x_S1, y_S1, x_S2, y_S2,x_S3, y_S3 = loadcifar()
            train = OLD3S_Deep(x_S1, y_S1, x_S2, y_S2,x_S3, y_S3, 50000, 5000,'parameter_cifar')

            train.train()

            print(time.time() - t)
        elif self.datasetname == 'svhn':
            print('svhn trainning starts')
            t = time.time()
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadsvhn()
            train = OLD3S_Deep(x_S1, y_S1, x_S2, y_S2, x_S3, y_S3, 73257, 7257, 'parameter_svhn')
            train.train()
            print(time.time() - t)
            #train = OLD3S_Deep(x_S1, y_S1, x_S2, y_S2, 73257, 7257,'parameter_svhn')
            #train.SecondPeriod()
        elif self.datasetname == 'mnist':
            print('mnist trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadmnist()
            if self.autoencoder == 'VAE':

                train = OLD3S_Mnist_VAE(x_S1, y_S1, x_S2, y_S2, 60000, 6000,dimension1 = 784, dimension2 = 784,
                                    hidden_size = 128, latent_size = 20, classes = 10, path = 'parameter_mnist')

                train.SecondPeriod()
            else:
                t = time.time()
                #train = OLD3S_Mnist(x_S1, y_S1, x_S2, y_S2, 60000, 6000, 'parameter_mnist')

                train = OLD3S_Mnist(x_S1, y_S1, x_S2, y_S2, x_S3, y_S3, 60000, 6000, 'parameter_mnist')
                train.train()
                print(time.time() - t)
        elif self.datasetname == 'magic':
            print('magic trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadmagic()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, x_S3, y_S3, 19019, 1919, 10, 30,50, 'parameter_magic')
            train.train()
        elif self.datasetname == 'adult':
            print('adult trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadadult()
            #train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 32559, 3559, 14, 30, 'parameter_adult')
            train = OLD3S_Shallow( x_S2, y_S2, x_S3, y_S3,x_S1, y_S1,32559, 3559,  30, 40, 14,'parameter_magic')
            train.train()

        elif self.datasetname == 'enfr':
            print('reuter-en-fr trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadreuter('EN_FR')

            if self.autoencoder == 'VAE':
                train = OLD3S_Shallow_VAE(x_S1, y_S1, x_S2, y_S2, 18758, 2758,2000, 2500,
                                    hidden_size = 128, latent_size = 20, classes = 6, path = 'parameter_enfr')
            else:

                train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2,x_S3, y_S3, 18758, 2758,21531,24893, 15506, 'parameter_enfr')

            train.train()
        elif self.datasetname == 'enit':
            print('reuter-en-it trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadreuter('EN_IT')
            print(x_S3.shape)
            #train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 18758, 2758, 2000, 1500, 'parameter_enit')
            train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, x_S3, y_S3, 18758, 2758, 21531, 15506, 11547, 'parameter_enit')
            train.train()
        elif self.datasetname == 'ensp':
            print('reuter-en-sp trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadreuter('EN_SP')
            #train = OLD3S_Reuter(x_S, y_S1, x_S2, y_S2, 18758, 2758, 2000, 1000, 'parameter_ensp')
            #train.SecondPeriod()
            train = OLD3S_Reuter(  x_S1, y_S1,x_S2, y_S2, x_S3, y_S3,18758, 2758,   21531,11547,24893,'parameter_ensp')
            train.train()
        elif self.datasetname == 'frit':
            print('reuter-fr-it trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadreuter('FR_IT')
            #train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 26648, 3648, 2500, 1500, 'parameter_frit')
            #train.SecondPeriod()
            train = OLD3S_Reuter(x_S1, y_S1,x_S2, y_S2, x_S3, y_S3, 26648,  3648,24893, 15503, 11547,  'parameter_frit')
            train.train()
        elif self.datasetname == 'frsp':
            print('reuter-fr-sp trainning starts')
            x_S1, y_S1, x_S2, y_S2, x_S3, y_S3 = loadreuter('FR_SP')
            #train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 26648, 3648, 2500, 1000, 'parameter_frsp')
            #train.SecondPeriod()
            train = OLD3S_Reuter(x_S1, y_S1,x_S2, y_S2, x_S3, y_S3,  26648, 3648, 24893,11547, 15503,   'parameter_frsp')
            train.train()
        else:
            print('Choose a correct dataset name please')

if __name__ == '__main__':
    setup_seed(30)
    main()





