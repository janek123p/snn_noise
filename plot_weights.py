from secrets import choice
import numpy as np
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
import os

plt.switch_backend('Agg')

def show_weights(filename, savename, wmax, arr = None, fignum = 1223, is_mnist = True):
    if arr is None:
        arr = np.load(filename)[:,2]
    weight_mat = arr.reshape(28*28 if is_mnist else 32*32*3,-1)
    
    n = int(np.sqrt(weight_mat.shape[1]))
    if is_mnist:
        rearranged_mat = np.zeros((28*n, 28*n))
        for i in range(weight_mat.shape[1]):
            rearranged_mat[int(i/n)*28:int(i/n+1)*28, int(i%n)*28:int(i%n+1)*28] = weight_mat[:,i].reshape(28,28)
    else:
        rearranged_mat = np.zeros((32*n, 32*n, 3))
        for i in range(weight_mat.shape[1]):
            red = weight_mat[:1024,i].reshape(32,32,1)
            green = weight_mat[1024:2*1024,i].reshape(32,32,1)
            blue = weight_mat[2*1024:3*1024,i].reshape(32,32,1)
            rearranged_mat[int(i/n)*32:int(i/n+1)*32, int(i%n)*32:int(i%n+1)*32, :] = np.concatenate((red, green, blue), axis=2)

    plt.figure(num = fignum, dpi=600)
    if is_mnist:
        plt.imshow(rearranged_mat, interpolation="nearest", cmap = cm.get_cmap('hot_r'), aspect='equal', extent = [0, n,0,n], vmin=0., vmax=wmax)
        plt.colorbar()
    else:
        plt.imshow(rearranged_mat, extent = [0, n,0,n])
    plt.title("Gewichte")
    plt.xticks(np.arange(0.5,n,1),range(1,n+1),fontsize = 7)
    plt.yticks(np.arange(0.5,n,1),range(1,n+1),fontsize = 7)
    plt.savefig(savename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Script to plot weight matrix.''')
    parser.add_argument('-label', dest='label', type=str, help='Name of the root directory of the simulation', required = True)
    parser.add_argument('-ending', dest = 'ending', type = str, help = 'Ending of the filename (e.g. 1000)', default='')
    parser.add_argument('-wmax', dest = 'wmax', type = float, help = 'Maximum weight', default=1.)
    parser.add_argument('-input', dest = 'input', type = str, help = 'mnist or cifar10', default='mnist', choices=['mnist', 'cifar10'])

    args = parser.parse_args(sys.argv[1:])
    label = args.label
    ending = args.ending
    path = '/mnt/data4tb/paessens/simulations/%s' % label
    is_mnist = args.input == 'mnist'

    if not os.path.exists(path):
        raise Exception('Directory %s does not exist!' %  (path))

    show_weights(path+'/weights/XeAe%s.npy' % ending, path + '/plots/XeAe%s.png' % ending, wmax = args.wmax, is_mnist = is_mnist)