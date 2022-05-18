import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='''Script to plot in-sample performance course''')
parser.add_argument('-label', dest='label', type=str, help='Name of the root directory of the simulation', required = True)
parser.add_argument('-performance_interval', dest='performance_interval', type=str, help='Interval the performance has been calculated', default = 500)
parser.add_argument('-mode', dest='mode', type=str, choices=["test", "train", "TEST", "TRAIN"], help='Test or training mode [Training]', default = "test")

args = parser.parse_args(sys.argv[1:])

label = args.label
mode = args.mode.lower()
distance = args.performance_interval

performance = np.load('/mnt/data4tb/paessens/simulations/%s/meta/performance_%s.npy' % (label, mode))

plt.plot(distance * np.arange(1,performance.shape[0]+1e-3, 1), performance)
plt.savefig('/mnt/data4tb/paessens/simulations/%s/plots/performance_%s.png' % (label, mode),dpi = 600)