from fileinput import filename
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='''Script to plot a mnist image''')
parser.add_argument('-filename', dest='filename', type=str, help='Path of the mnist image', required = True)
args = parser.parse_args(sys.argv[1:])

mnist_arr = np.load(args.filename).reshape(28,28)

min = mnist_arr.min()
max = mnist_arr.max()

plt.imshow(mnist_arr, interpolation="nearest", cmap = cm.get_cmap('hot_r'), aspect='equal', extent = [0, 28,0,28], vmin=min, vmax=max)
plt.colorbar()
plt.title("Mnist image %s" % args.filename)
plt.savefig('mnist.png')