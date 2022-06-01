'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
from brian2 import *
import sys
import argparse
import os

from sklearn.svm import SVC 
from functions.data import get_labeled_data

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments

def train_SVM(X, y):
    print("Training SVM...")
    svc = SVC()
    svc.fit(X, y)
    return svc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Script to initialize the directory structure for a simulation including generating random weights ''')
    parser.add_argument('-label', dest='label', type=str, help='Name of the root directory of the directory strucuture that is created', required = True)
    parser.add_argument('-num_assigns', dest='assignment_number', type=int, help='Number of training results that are used to calculate assignments', default = 10000)
    parser.add_argument('-datapath', dest='data_path', type=str, help='Datapath to load MNIST-images from', default = './mnist/')
    parser.add_argument('-test_label', dest='test_label', type=str, help='Label to identify test case.', default = None)
    parser.add_argument('-svm', dest='svm', help='Whether to use SVM to classify output', action = "store_true")
    parser.add_argument('-N', dest='N', type=int, help='Number of inhibitory and excitatory neurons', default = 400)


    args = parser.parse_args(sys.argv[1:])
    MNIST_data_path = args.data_path
    label = args.label
    test_label = args.test_label
    num_assign = args.assignment_number
    svm = args.svm
    path = '/mnt/data4tb/paessens/simulations/%s' % label

    if not os.path.exists(path):
        raise Exception("No directory (%s) corresponding to the given label does exist!" % path)

    data_path = path + '/activity/'

    n_e = args.N # Number of excitatory neurons
    n_input = 784 # Number of input neurons
    ending = ''

    print('Loading MNIST dataset...')
    training = get_labeled_data(MNIST_data_path + 'training', MNIST_data_path=MNIST_data_path + 'training')
    testing = get_labeled_data(MNIST_data_path + 'training', MNIST_data_path=MNIST_data_path + 'testing', bTrain = False)

    print('Loading simulation results...')
    training_result_monitor = np.load(data_path + 'resultPopVecs_train.npy')[-num_assign:]
    training_input_numbers = np.load(data_path + 'inputNumbers_train.npy')[-num_assign:]
    if test_label is None:
        testing_result_monitor = np.load(data_path + 'resultPopVecs_test.npy')
        testing_input_numbers = np.load(data_path + 'inputNumbers_test.npy')
    else:
        testing_result_monitor = np.load(data_path + 'resultPopVecs_test_%s.npy' % test_label)
        testing_input_numbers = np.load(data_path + 'inputNumbers_test_%s.npy' % test_label)

    #print(training_result_monitor.shape)
    test_size = len(testing_input_numbers)
    test_results = np.zeros(test_size)
    if not svm:
        print('Calculating assignments...')
        assignments = get_new_assignments(training_result_monitor, training_input_numbers)
        print('Calculating accuracy...')
        for i in range(test_size):
            test_results[i] = get_recognized_number_ranking(assignments, testing_result_monitor[i,:])[0]
    else:
        svc = train_SVM(training_result_monitor, training_input_numbers)
        print("Calculating predictions...")
        test_results = svc.predict(testing_result_monitor)


    difference = test_results[:] - testing_input_numbers[:]
    correct_indices = np.where(difference == 0)[0]
    incorrect_indices = np.where(difference != 0)[0]
    correct = len(correct_indices)
    incorrect = len(incorrect_indices)
    print('Accuracy: %.3f, Correct classified: %d/%d, Incorrect classified %d/%d' % (correct / test_size, correct, test_size, incorrect, test_size))

    plt.figure(num=212)

    sqrt_incorrect = np.sqrt(incorrect)
    n_cols = int(sqrt_incorrect)
    if n_cols * n_cols != incorrect:
        n_cols += 1
    
    wrong_images = np.zeros((n_cols*28, n_cols*28))
    for i, idx in enumerate(incorrect_indices):
        x = i % n_cols
        y = i // n_cols
        wrong_images[x*28:(x+1)*28, y*28:(y+1)*28] = testing['x'][idx].reshape(28,28)

    plt.imshow(wrong_images, interpolation="nearest", cmap = 'gray' , aspect='equal', extent = [0, n_cols,0,n_cols])
    plt.colorbar()
    plt.title("Wrong classified MNIST images")
    plt.savefig(path + '/plots/wrong_classified.png', dpi = 600)

    plt.figure(num=213)

    classification_matrix = np.zeros((10,10))
    for i in range(test_size):
        desired = testing_input_numbers[i]
        output = int(test_results[i])
        classification_matrix[desired, output] += 1

    np.save(path+'/meta/classification_matrix_%s.npy' % ('std' if test_label is None else test_label), classification_matrix)

    for i in range(10):
        classification_matrix[i,:] /= np.sum(classification_matrix[i,:])


    plt.imshow(np.flip(classification_matrix.T, axis = 0), interpolation="nearest", cmap = cmap.get_cmap('hot_r'), aspect='equal', extent = [-0.5, 9.5,-0.5,9.5])
    plt.colorbar()
    plt.xlabel('Desired output')
    plt.ylabel('Prediction')
    plt.xticks(np.arange(0,10,1))
    plt.yticks(np.arange(0,10,1))

    for i in range(10):
        for j in range(10):
            val = int(np.round(classification_matrix[i,j]*100))
            plt.text(i-0.4,j-0.15,"%2d%%" % val, color ='white' if val > 50 else 'black' )

    plt.savefig(path + '/plots/classification_matrix.png', dpi = 600)

    print('Evaluation done!')