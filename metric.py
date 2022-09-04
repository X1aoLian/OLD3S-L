import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import *
from scipy.interpolate import make_interp_spline
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_reuter(y_axi_1,y_axi_2, x, path, a, b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(range(20))
    ax.axvspan(b - a, b, alpha=0.5, color='#86C6F4')
    ax.axvspan(2*b - a, 2*b, alpha=0.5, color='#86C6F4')
    plt.grid()
    x_smooth = np.linspace(x.min(), x.max(), 25)
    y_smooth_1 = make_interp_spline(x, y_axi_1)(x_smooth)
    y_smooth_2 = make_interp_spline(x, y_axi_2)(x_smooth)
    ACR(y_smooth_1,y_smooth_2)
    STD(25, y_axi_1, y_smooth_1)
    ACR(y_smooth_2,y_smooth_1)
    STD(25, y_axi_2, y_smooth_2)
    plt.plot(x_smooth, y_smooth_2, color='#7E2F8E', marker='o',label='OLD3S' )
    plt.plot(x_smooth, y_smooth_1, color='#332FD0', marker='^',label='OLD3S-L')
    ax.set_xlim(250, b - a + b + b)
    ax.set_ylim(0.8,1)
    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('# of instances', fontsize=30)
    plt.ylabel('OCA', fontsize=30)
    plt.legend(fontsize = 'xx-large', loc = 'upper left')
    plt.tight_layout()
    plt.savefig(path)


def ACR(accuracy1, accuracy2):
    f_star1 = max(accuracy1)
    f_star2 = max(accuracy2)
    f_star = max(f_star1,f_star2)
    acr = mean([f_star - i for i in accuracy1])
    print(acr)


def STD(filternumber, elements,smoothlist):
    gap = len(elements)//filternumber
    std = 0
    for i in range(filternumber):
        for j in range(int(gap)):
            std += np.abs(smoothlist[i] - elements[i*gap: (i+1)*gap][j])
    print(std/len(elements))


x = np.array([i for i in range(1000, 3*18758 - 2758 , 500)])
print(x.shape)
path = 'D:/pycharmproject/OLD3S/model/data/enfr.png'
y_axi_1 = np.array(torch.load('./data/parame_enfr/Accuracy')).tolist()
print(len(y_axi_1))
y1 = torch.load('./data/parame_enfr/Accuracy_S')
print(len(y1))
y2 = torch.load('./data/parame_enfr/Accuracy_S2')[32:]#32
print(len(y2))
y_axi_2 = y1 + y2
#plot_reuter(y1, x, path,  2758 ,18758)
plot_reuter(y_axi_1,y_axi_2, x, path, 2758 ,18758)

'''x = np.array([i for i in range(500, 3*50000 - 5000 , 1000)])
print(x.shape)
path = 'D:/pycharmproject/OLD3S/model/data/cifar1.png'
y_axi_1 = np.array(torch.load('./data/parameter_cifar/Accuracy')).tolist()
print(len(y_axi_1))
y1 = torch.load('./data/parameter_cifar/Accuracy_S')
print(len(y1))
y2 = torch.load('./data/parameter_cifar/Accuracy_S2')[45:]#32
print(len(y2))
y_axi_2 = y1 + y2
#plot_reuter(y1, x, path,  2758 ,18758)
plot_reuter(y_axi_1,y_axi_2, x, path, 5000 ,50000)'''



'''y1 = torch.load('./data/parameter_mnist/Accuracy_S')
y2 = torch.load('./data/parameter_mnist/Accuracy_S2')[54:]
y_axi_2 = y1 + y2
y_axi_1 = np.array(torch.load('./data/parameter_mnist/Accuracy')).tolist()
print(len(y1))
print(len(y2))
print(len(y_axi_2))
print(len(y_axi_1))

error_1 = 0
error_2 = 0
for i in y_axi_1:
    error_1 += (1000 - i * 1000)
print(error_1)
for j in y_axi_2:
    error_2 += (1000 - j * 1000)
print(error_2)'''

