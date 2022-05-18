import numpy as np
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
import os

plt.switch_backend('Agg')

def show_weights(filename, savename, wmax, arr = None, fignum = 1223):
    if arr is None:
        weights = np.load(filename)
        weight_mat = weights[:,2].reshape(28*28,-1)
    else:
        weight_mat = arr.reshape(28*28, -1)
    n_ex = int(np.sqrt(weight_mat.shape[1]))
    rearranged_mat = np.zeros((28*n_ex, 28*n_ex))
    for i in range(weight_mat.shape[1]):
        rearranged_mat[int(i/n_ex)*28:int(i/n_ex+1)*28, int(i%n_ex)*28:int(i%n_ex+1)*28] = weight_mat[:,i].reshape(28,28)

    plt.figure(num = fignum)
    plt.imshow(rearranged_mat, interpolation="nearest", cmap = cm.get_cmap('hot_r'), aspect='equal', extent = [0, 20,0,20], vmin=0., vmax=wmax)
    plt.colorbar()
    plt.title("Rearranged weights")
    plt.savefig(savename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Script to plot weight matrix.''')
    parser.add_argument('-label', dest='label', type=str, help='Name of the root directory of the simulation', required = True)
    parser.add_argument('-ending', dest = 'ending', type = str, help = 'Ending of the filename (e.g. 1000)', default='')
    parser.add_argument('-wmax', dest = 'wmax', type = float, help = 'Maximum weight', default=1.)
    args = parser.parse_args(sys.argv[1:])
    label = args.label
    ending = args.ending
    path = '/mnt/data4tb/paessens/simulations/%s' % label

    if not os.path.exists(path):
        raise Exception('Directory %s does not exist!' %  (path))

    show_weights(path+'/weights/XeAe%s.npy' % ending, path + '/plots/XeAe%s.png' % ending, wmax = args.wmax)