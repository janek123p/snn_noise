import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='''Script to plot in-sample performance course''')
parser.add_argument('-label', dest='label', type=str, help='Name of the root directory of the simulation', required = True)
parser.add_argument('-performance_interval', dest='performance_interval', type=str, help='Interval the performance has been calculated', default = 500)
args = parser.parse_args(sys.argv[1:])

label = args.label
distance = args.performance_interval

performance = np.load('./simulations/%s/meta/performance.npy' % label)

plt.plot(distance * np.arange(1,performance.shape[0]+1e-3, 1), performance)
plt.savefig('./simulations/%s/plots/performance.png' % label,dpi = 600)