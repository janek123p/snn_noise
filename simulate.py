'''
Created on 15.12.2014
by P. U. Diehl

modified by Janek Paessens
'''

# imports
import sys
import argparse
import os
import math
import random
import numpy as np
import scipy.stats as stat
import time
from brian2 import *
import os
import brian2 as b2
from brian2tools import *
from global_settings import settings

# imports from own modules
from functions.data import get_labeled_data
from functions.input_noise import salt_and_pepper, remove_rectangle, plot_salt_and_pepper_examples, plot_remove_rectangle_examples


######################################################
# ARGUMENT PARSING                                   #
######################################################

parser = argparse.ArgumentParser(description='''Simulation of SNN to classify MNIST images based on\
[https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full]. \
Original code can be found here: [https://github.com/peter-u-diehl/stdp-mnist].\
Migration of the original code to Brian2 and Python 3 can be found here: [https://github.com/sdpenguin/Brian2STDPMNIST].''')
parser.add_argument('-mode', dest='mode', type=str, choices=[
                    'test', 'train', 'TEST', 'TRAIN', 'training', 'TRAINING'], help='Either test or training', required=True)
parser.add_argument('-label', dest='path', type=str,
                    help='Label to save output files with', required=True)
parser.add_argument('-data', dest='datapath', type=str,
                    help='Data path of MNIST or cifar dataset [./mnist/]', default='mnist/')
parser.add_argument('-epochs', dest='epochs', type=int,
                    help='Number of epochs to train data with [1]', default=1)
parser.add_argument('-train_size', dest='train_size', type=int,
                    help='Number of inputs to train data each epoch with [60000]', default=60000)
parser.add_argument('-test_size', dest='test_size', type=int,
                    help='Number of inputs to test data with [10000]', default=10000)
parser.add_argument('-plasticity', dest='plast_during_testing', help='Whether the network ist static\
or plastic during testing phase', action='store_true')
parser.add_argument('-synapse_model', dest='syn_model', choices=['triplet', 'clopath', 'TRIPLET', 'CLOPATH'],
                    type=str, help='Whether triplet stdp or clopath stdp should be used [triplet]', default='triplet')
parser.add_argument('-debug', dest='debug',
                    help='Whether debug information should be printed, plotted and saved. CAUTION: May by storage-consuming! [False]', action='store_true')
parser.add_argument('-test_label', dest='test_label',
                    help='Label to identify test cas with [None]', default=None, type=str)
parser.add_argument(
    '-N', dest='N', help='Number of excitatory and inhibitory neurons [400]', default=400, type=int)
parser.add_argument('-input', dest='input_type',
                    help='mnist or cifar10 [mnist]', default="mnist", type=str, choices=["mnist", "cifar10"])


parser.add_argument('-rand_threshold_max', dest='rand_thresh_max', type=float,
                    help='Maximal value of random threshold in mV [0mV]', default=0.)
parser.add_argument('-rand_threshold_min', dest='rand_thresh_min',
                    type=float, help='Minimal value of random threshold [0mV]', default=0.)
parser.add_argument('-noise_membrane_voltage_max', dest='noise_membrane_voltage_max', type=float, help='Maximal value of random adjustment\
of membrane voltage per timestep for excitatory neurons [0mV]', default=0.)
parser.add_argument('-noise_membrane_voltage_min', dest='noise_membrane_voltage_min', type=float, help='Minimal value of random adjustment\
of membrane voltage per timestep for excitatory neurons [0mV]', default=0.)
parser.add_argument('-voltage_noise_sigma', dest='sigma_v', type=float,
                    help='Standard deviation of the noise added to the membrane voltage of excitatory neurons every timestep in mV [0]', default=0.)
parser.add_argument('-voltage_noise_sigma_inh', dest='sigma_v_inh', type=float,
                    help='Standard deviation of the noise added to the membrane voltage of inhibitory neurons every timestep in mV [0]', default=0.)
parser.add_argument('-membrane_voltage_quant', dest='membrane_voltage_quant', type=int,
                    help='Number of bits to quantify membane voltage of excitatory neurons [None]', default=None)
parser.add_argument('-membrane_voltage_quant_inh', dest='membrane_voltage_quant_inh', type=int,
                    help='Number of bits to quantify membane voltage of inhibitory neurons [None]', default=None)
parser.add_argument('-weight_quant', dest='weight_quant', type=int,
                    help='Number of bits to quantify weights [None]', default=None)
parser.add_argument('-stoch_weight_quant', dest='stoch_weight_quant', type=int,
                    help='Number of bits to quantify weights. Quantization is performed in a stochastic manner. [None]', default=None)
parser.add_argument('-salt_and_pepper_alpha', dest='salt_pepper_alpha', type=float,
                    help='Propability that a pixel in the input image gets replaced by 0 or 255 [None]', default=None)
parser.add_argument('-rectangle_noise_min', dest='rectangle_noise_min', type=int,
                    help='Minimal width and height of the rectangle that is removed from the input image [None]', default=None)
parser.add_argument('-rectangle_noise_max', dest='rectangle_noise_max', type=int,
                    help='Maximal width and height of the rectangle that is removed from the input image [None]', default=None)
parser.add_argument('-p_dont_send_spike', dest='p_dont_send_spike', type=float, help='Propability that a spike in excitatory layer occurs without increasing\
the postsynaptic conductance in the inhibitory layer [None]', default=None)
parser.add_argument('-p_dont_send_spike_inh', dest='p_dont_send_spike_inh', type=float, help='Propability that a spike in inhibitory layer occurs without increasing\
the postsynaptic conductance in the excitatory layer [None]', default=None)
parser.add_argument('-sigma_heterogenity', dest='sigma_het', type=float,
                    help='Standard deviation of neural heterogenity as proportion of the mean value [None]', default=None)


args = parser.parse_args(sys.argv[1:])

test_mode = args.mode.upper() == 'TEST'
filename_label = args.path
input_data_path = args.datapath
epochs = args.epochs
train_size = args.train_size
test_size = args.test_size
plasticity_during_testing = args.plast_during_testing
min_rand_theta = args.rand_thresh_min
diff_rand_theta = args.rand_thresh_max - min_rand_theta
noise_v_min = args.noise_membrane_voltage_min
noise_v_diff = args.noise_membrane_voltage_max - noise_v_min
sigma_v = args.sigma_v*b2.mV
sigma_v_inh = args.sigma_v_inh*b2.mV
sigma_het = args.sigma_het
v_quant = args.membrane_voltage_quant
v_quant_inh = args.membrane_voltage_quant_inh
w_quant = args.weight_quant
stoch_w_quant = args.stoch_weight_quant
salt_pepper_alpha = args.salt_pepper_alpha
rectangle_noise_min = args.rectangle_noise_min
rectangle_noise_max = args.rectangle_noise_max
p_dont_send_spike = args.p_dont_send_spike
p_dont_send_spike_inh = args.p_dont_send_spike_inh
clopath = args.syn_model.upper() == 'CLOPATH'
save_debug_info = args.debug
test_label = args.test_label
is_mnist = args.input_type == 'mnist'

if test_label is None:
    test_label = "std"

if w_quant is not None and stoch_w_quant is not None:
    raise Exception(
        "It can only be on of the following quantizations be active: Either weight_quant or stoch_weight_quant")

# if either max (or min) is None set max (min) to min (max)
# if both are None they stay None
if rectangle_noise_max is None:
    rectangle_noise_max = rectangle_noise_min
if rectangle_noise_min is None:
    rectangle_noise_min = rectangle_noise_max

if test_mode:
    print_addon = filename_label+"_"+test_label+": "
else:
    print_addon = filename_label+": "

data_path = settings['simulation_base_path']+filename_label + '/'
if not os.path.exists(data_path):
    raise Exception(
        print_addon+"Directory %s does not exist! Create it before running this script!" % data_path)

subfolder = ['plots', 'weights', 'activity', 'random', 'meta']
for subf in subfolder:
    if not os.path.exists(data_path+subf):
        raise Exception(
            print_addon+"Directory structure is incompletet! Directory %s is missing!" % (data_path+subf))

summary = 'test_mode = %s\nlabel = %s\nepochs = %d\ntrain size = %d\ntest size = %d\nplasticity during testing = %s\nSynapse model = %s\n' \
    % (test_mode, filename_label+"_"+test_label if test_mode else filename_label, epochs, train_size, test_size, plasticity_during_testing, 'clopath' if clopath else 'triplet')

summary += '\n'
if diff_rand_theta > 0:
    summary += 'Minimal random threshold = %.5f\nMaximal random thersold = %.5f\n' % (
        min_rand_theta, min_rand_theta + diff_rand_theta)
if noise_v_min > 0 and noise_v_diff > 0:
    summary += 'Minimal random adjustment of the membrane voltage = %.5f\nMaximal random adjustment of the membrane voltage %.5f\n' % (
        noise_v_min, noise_v_min + noise_v_diff)
if v_quant is not None:
    summary += 'Number of bits to quantify membrane voltage of exc. neurons: %d\n' % v_quant
if v_quant_inh is not None:
    summary += 'Number of bits to quantify membrane voltage of inh. neurons: %d\n' % v_quant_inh
if w_quant is not None:
    summary += 'Number of bits to quantify weights: %d\n' % w_quant
if stoch_w_quant is not None:
    summary += 'Number of bits to quantify weights stochastically: %d\n' % stoch_w_quant
if salt_pepper_alpha is not None:
    summary += 'Propability of salt or pepper in input image: %.4f\n' % salt_pepper_alpha
if rectangle_noise_max is not None:
    summary += 'Minimal width of removed rectangle: %d\nMinimal width of removed rectangle: %d\n' % (
        rectangle_noise_min, rectangle_noise_max)
if p_dont_send_spike is not None:
    summary += 'Propability that a presynaptic spike in the excitatory layer does not lead to an increase of the postsynaptic conductance: %.4f\n' % (
        p_dont_send_spike)
if p_dont_send_spike_inh is not None:
    summary += 'Propability that a presynaptic spike in the inhibitory layer does not lead to an increase of the postsynaptic conductance: %.4f\n' % (
        p_dont_send_spike_inh)
if abs(sigma_v/b2.mV) > 1e-10:
    summary += 'Normally distributed noise of the membrane voltage of exc. neurons: %.6f mV/dt\n' % (
        sigma_v / b2.mV)
if abs(sigma_v_inh/b2.mV) > 1e-10:
    summary += 'Normally distributed noise of the membrane voltage of inh. neurons: %.6f mV/dt\n' % (
        sigma_v_inh / b2.mV)
if sigma_het is not None:
    summary += 'Standard deviation of neural heterogenity as proportion of the mean value: %.3f \n' % sigma_het


b2.prefs.codegen.target = 'cython'
b2.defaultclock.dt = 0.5 * b2.ms

# ------------------------------------------------------------------------------
# functions
# ------------------------------------------------------------------------------


def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset] == 'e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset] == 'e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(
            readout[:, 1])] = readout[:, 2]
    return value_arr


def save_connections(ending=''):
    print(print_addon+'Saving connections...')
    for connName in save_conns:
        conn = connections[connName]
        connListSparse = list(zip(conn.i, conn.j, conn.w))
        np.save(data_path + 'weights/' + connName + ending, connListSparse)


def save_theta(ending=''):
    print(print_addon+'Saving theta...')
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name +
                ending, neuron_groups[pop_name + 'e'].theta)


def stoch_rounding_arr(arr, round_val):
    arr2 = round_val * arr
    floor_arr = np.floor(arr2)
    return (floor_arr + ((arr2-floor_arr) > np.random.random(arr.shape))) / round_val


def normalize_weights():
    connName = 'XeAe'
    len_source = len(connections[connName].source)
    len_target = len(connections[connName].target)
    connection = np.zeros((len_source, len_target))
    connection[connections[connName].i,
               connections[connName].j] = connections[connName].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis=0)
    colFactors = weight['ee_input']/colSums
    for j in range(n_e):
        temp_conn[:, j] *= colFactors[j]
    if w_quant is not None:
        temp_conn = np.round(temp_conn * round_val_w) / round_val_w
    if stoch_w_quant is not None:
        temp_conn = stoch_rounding_arr(temp_conn, round_val_w)
    connections[connName].w = temp_conn[connections[connName].i,
                                        connections[connName].j]


def get_current_performance(current_example_num):
    start_num = current_example_num - calculate_performance_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num,
                               0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    return float(correct) / (end_num - start_num)


def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(
                spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(
                result_monitor[input_nums == j], axis=0) / num_assignments
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments


def print_progress(j, total, last, num_since_last):
    duration = time.time() - last
    print("%d of %d [%.2fs/it]" % (j, total, duration/num_since_last))


def quantile_sorted(sorted_arr, quantile):
    max_index = len(sorted_arr) - 1
    quantile_index = max_index * quantile
    quantile_index_int = int(quantile_index)
    quantile_index_fractional = quantile_index - quantile_index_int

    quantile_lower = sorted_arr[quantile_index_int]
    if quantile_index_fractional > 0:
        quantile_upper = sorted_arr[quantile_index_int + 1]
        return quantile_lower + (quantile_upper - quantile_lower) * quantile_index_fractional
    else:
        return quantile_lower

# ------------------------------------------------------------------------------
# load MNIST
# ------------------------------------------------------------------------------


print("Loading dataset...")
if is_mnist:
    start = time.time()
    training = get_labeled_data(
        input_data_path + 'training', MNIST_data_path=input_data_path)
    end = time.time()
    print(print_addon+'Time needed to load training set: %.4fs' % (end - start))

    start = time.time()
    testing = get_labeled_data(
        input_data_path + 'testing', MNIST_data_path=input_data_path, bTrain=False)
    end = time.time()
    print(print_addon+'Time needed to load test set: %.4fs' % (end - start))
else:
    import pickle
    training = {'x': np.zeros((50000, 3072), dtype=np.ubyte),
                'y': np.zeros((50000,), dtype=np.ubyte)}
    testing = {'x': np.zeros((10000, 3072), dtype=np.ubyte),
               'y': np.zeros((10000,), dtype=np.ubyte)}
    for i in range(5):
        with open(input_data_path+'/data_batch_%d' % (i+1), 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
            training['x'][i*10000:(i+1)*10000,
                          :] = np.array(dict[b'data'], dtype=np.ubyte)
            training['y'][i *
                          10000:(i+1)*10000] = np.array(dict[b'labels'], dtype=np.ubyte)
    with open(input_data_path+'/test_batch', 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        testing['x'][:, :] = np.array(dict[b'data'], dtype=np.ubyte)
        testing['y'][:] = np.array(dict[b'labels'], dtype=np.ubyte)


# ------------------------------------------------------------------------------
# set parameters and equations
# ------------------------------------------------------------------------------

if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = test_size
    ee_STDP_on = plasticity_during_testing
else:
    weight_path = data_path + 'random/'
    num_examples = train_size * epochs
    ee_STDP_on = True

if save_debug_info:
    save_connections_interval = 100
    calculate_performance_interval = 50
else:
    save_connections_interval = 1000
    calculate_performance_interval = 500

print_progress_interval = max(int(num_examples / 500), 1)

ending = ''
if is_mnist:
    n_input = 784
else:
    n_input = 32*32*3
n_e = args.N
n_i = n_e
single_example_time = 0.35 * b2.second
resting_time = 0.15 * b2.second
runtime = num_examples * (single_example_time + resting_time)

v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b2.ms, 10*b2.ms)
delay['ei_input'] = (0*b2.ms, 5*b2.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 40*b2.ms
nu_ee_pre = 0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4
offset = 20.0*b2.mV
tau = 100 * b2.ms
tau_ge = 1 * b2.ms
tau_gi = 2 * b2.ms

if w_quant is not None:
    round_val_w = 2**w_quant
if stoch_w_quant is not None:
    round_val_w = 2**stoch_w_quant
if v_quant is not None:
    round_val_v = 2**v_quant
if v_quant_inh is not None:
    round_val_v_inh = 2**v_quant_inh


@b2.implementation('cython', '''
    cdef double round_val(double x,int val):
        return round(x*val)/val
    ''')
@b2.check_units(x=1, val=1, result=1)
def round_val(x, val):
    return np.round(x*val)/val


@b2.implementation('cython', '''
    from random import random

    cdef stoch_rounding(double x, int val):
        value = x * val
        floor_val = int(value)
        return (floor_val + ((value - floor_val) > random())) / val
    ''')
@b2.check_units(x=1, val=1, result=1)
def stoch_rounding(x, val):
    value = x * val
    floor_val = math.floor(val)
    return (floor_val + ((value - floor_val) > random.random())) / val


if test_mode and not plasticity_during_testing:
    if v_quant is not None:
        reset_e_str = 'x = v_reset_e; timer = 0*ms'
    else:
        reset_e_str = 'v = v_reset_e; timer = 0*ms'
else:
    theta_plus_e = 0.05 * b2.mV

    if n_e <= 400 and not clopath:
        tc_theta = 1e7 * b2.ms
    elif n_e <= 400 and clopath:
        tc_theta = 3e7 * b2.ms
    elif n_e > 400:
        tc_theta = 5e7 * b2.ms
    elif n_e > 1600:
        tc_theta = 1e8 * b2.ms
        theta_plus_e = 0.075 * b2.mV

    summary += "\ntc_theta = %.5Ems\ntheta_plus_e = %.5EmV\n" % (
        tc_theta/b2.ms, theta_plus_e/b2.mV)

    if v_quant is not None:
        reset_e_str = 'x = v_reset_e; theta += theta_plus_e; timer = 0*ms'
    else:
        reset_e_str = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

v_thresh_e_str = '(v > (rand_theta + theta - offset + v_thresh_e)) and (timer>refrac_e)'

reset_e_str += '; rand_theta = min_rand_theta*mV + rand()*diff_rand_theta*mV'

v_thresh_i_str = 'v>v_thresh_i'
if v_quant_inh is None:
    v_reset_i_str = 'v=v_reset_i'
else:
    v_reset_i_str = 'x=v_reset_i'


if v_quant is not None:
    neuron_eqs_e = '''
            dx/dt = ((v_rest_e - x) + (I_synE+I_synI) / nS) / tau + (noise_v_min*mV + rand()*noise_v_diff * mV)/dt + sigma_v*sqrt(dt)*xi/dt  : volt (unless refractory)
            v = round_val(x/mV , round_val_v) * mV                  : volt 
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-100.*mV-v) * int(v > -100.*mV)     : amp
            dge/dt = -ge/tau_ge                                     : 1
            dgi/dt = -gi/tau_gi                                     : 1
            dtimer/dt = 1                                           : second
            rand_theta                                              : volt
            '''
else:
    neuron_eqs_e = '''
            dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / tau + (noise_v_min*mV + rand()*noise_v_diff * mV)/dt + sigma_v*sqrt(dt)*xi/dt : volt (unless refractory)
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-100.*mV-v) * int(v > -100.*mV)     : amp
            dge/dt = -ge/tau_ge                                     : 1
            dgi/dt = -gi/tau_gi                                     : 1
            dtimer/dt = 1                                           : second
            rand_theta                                              : volt
            '''

if sigma_het is not None:
    neuron_eqs_e += '''\ntau : second
                        tau_ge : second
                        tau_gi : second
                        refrac_e : second'''

if test_mode and not plasticity_during_testing:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'


if clopath:
    tau_minus = 40*b2.ms
    tau_plus = 30*b2.ms
    neuron_eqs_e += '\n du_minus/dt = (v - u_minus) / tau_minus   : volt'
    neuron_eqs_e += '\n du_plus/dt = (v - u_plus) / tau_plus      : volt'

if v_quant_inh is None:
        neuron_eqs_i = '''
            dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) + sigma_v*sqrt(dt)*xi/dt  : volt (unless refractory)
            I_synE = ge * nS *         -v                                                       : amp
            I_synI = gi * nS * (-85.*mV-v)                                                      : amp
            dge/dt = -ge/(1.0*ms)                                                               : 1
            dgi/dt = -gi/(2.0*ms)                                                               : 1
            '''
else:
    neuron_eqs_i = '''
            dx/dt  = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) + sigma_v*sqrt(dt)*xi/dt  : volt (unless refractory)
            v      = round_val(x/mV, round_val_v_inh) * mV                                       : volt
            I_synE = ge * nS *         -v                                                        : amp
            I_synI = gi * nS * (-85.*mV-v)                                                       : amp
            dge/dt = -ge/(1.0*ms)                                                                : 1
            dgi/dt = -gi/(2.0*ms)                                                                : 1
            '''



if not clopath:
    eqs_stdp_ee = '''
                    post2before                                     : 1
                    dpre/dt   =   -pre/(tc_pre_ee)                  : 1 (event-driven)
                    dpost1/dt  = -post1/(tc_post_1_ee)              : 1 (event-driven)
                    dpost2/dt  = -post2/(tc_post_2_ee)              : 1 (event-driven)
                '''
    if w_quant is not None:
        eqs_stdp_pre_ee = 'pre = 1.; w = round_val(clip(w - nu_ee_pre * post1, 0, wmax_ee), round_val_w)'
        eqs_stdp_post_ee = 'post2before = post2; w = round_val(clip(w + nu_ee_post * pre * post2before, 0, wmax_ee), round_val_w); post1 = 1.; post2 = 1.'
    elif stoch_w_quant is not None:
        eqs_stdp_pre_ee = 'pre = 1.; w = stoch_rounding(clip(w - nu_ee_pre * post1, 0, wmax_ee), round_val_w)'
        eqs_stdp_post_ee = 'post2before = post2; w = stoch_rounding(clip(w + nu_ee_post * pre * post2before, 0, wmax_ee), round_val_w); post1 = 1.; post2 = 1.'
    else:
        eqs_stdp_pre_ee = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
        eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'
else:
    tau_x = 15 * b2.ms
    A_LTD = 1e-9/b2.mV  # 1e-6
    A_LTP = 1e-4/(b2.mV*b2.mV)  # 1.5*1e-4
    x_reset = 1*b2.ms
    summary += "A_LTD = %.5E\nA_LTP = %.5E\n" % (
        A_LTD*b2.mV, A_LTP*b2.mV*b2.mV)
    theta_minus = v_reset_e
    theta_plus = v_reset_e
    eqs_stdp_ee = '''
                    dxpre/dt   =   -xpre/(tau_x)         : 1 (event-driven)
                '''

    # POTENTIATION ONLY WORKS IF theta_plus = v_reset_e ==> OTHERWISE THE EQUATIONS BELOW ARE INCORRECT
    if w_quant is not None:
        eqs_stdp_pre_ee = 'xpre = xpre + x_reset/tau_x; w = round_val(clip(w - A_LTD * (u_minus_post - theta_minus) * int(u_minus_post > theta_minus), 0, wmax_ee), round_val_w)'
        eqs_stdp_post_ee = 'w = round_val(clip(w + A_LTP * xpre * (v_post - theta_plus) * int(v_post > theta_plus) * (u_plus_post - theta_minus) * int(u_plus_post > theta_minus), 0, wmax_ee) , round_val_w) '
    elif stoch_w_quant is not None:
        eqs_stdp_pre_ee = 'xpre = xpre + x_reset/tau_x; w = stoch_rounding(clip(w - A_LTD * (u_minus_post - theta_minus) * int(u_minus_post > theta_minus), 0, wmax_ee), round_val_w)'
        eqs_stdp_post_ee = 'w = stoch_rounding(clip(w + A_LTP * xpre * (v_post - theta_plus) * int(v_post > theta_plus) * (u_plus_post - theta_minus) * int(u_plus_post > theta_minus), 0, wmax_ee) , round_val_w) '
    else:
        eqs_stdp_pre_ee = 'xpre = xpre + x_reset/tau_x; w = clip(w - A_LTD * (u_minus_post - theta_minus) * int(u_minus_post > theta_minus), 0, wmax_ee)'
        eqs_stdp_post_ee = 'w = clip(w + A_LTP * xpre * (v_post - theta_plus) * int(v_post > theta_plus) * (u_plus_post - theta_minus) * int(u_plus_post > theta_minus), 0, wmax_ee)'


b2.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
spike_counters = {}
result_monitor = np.zeros((int(num_examples), n_e))

neuron_groups['Ae'] = b2.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold=v_thresh_e_str,
                                     refractory=refrac_e if sigma_het is None else 'refrac_e', reset=reset_e_str, method='euler')
neuron_groups['Ai'] = b2.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold=v_thresh_i_str,
                                     refractory=refrac_i, reset=v_reset_i_str, method='euler')

# ------------------------------------------------------------------------------
# create network population and recurrent connections
# ------------------------------------------------------------------------------
for subgroup_n, name in enumerate(population_names):
    print(print_addon+'Creating neuron group %s...' % name)

    if v_quant is None:
        neuron_groups[name+'e'].v = v_rest_e - 40. * b2.mV
    else:
        neuron_groups[name+'e'].x = np.round(
            (v_rest_e/b2.mV - 40.) * round_val_v) / round_val_v*b2.mV

    if v_quant_inh is None:
        neuron_groups[name+'i'].v = v_rest_i - 40. * b2.mV
    else:
        neuron_groups[name+'i'].x = np.round(
            (v_rest_i/b2.mV - 40.) * round_val_v_inh) / round_val_v_inh*b2.mV

    if sigma_het is not None:
        def truncated_normal_values(mean, std_proportion, lower, upper, n):
            X = stat.truncnorm((lower-mean) / (std_proportion * mean), (upper-mean) / (
                std_proportion * mean), loc=mean, scale=std_proportion * mean)
            return np.array(X.rvs(n))

        neuron_groups[name+'e'].tau = truncated_normal_values(
            tau/b2.ms, sigma_het, 20, 1e9, n_e) * b2.ms
        neuron_groups[name+'e'].tau_ge = truncated_normal_values(
            tau_ge/b2.ms, sigma_het, 0.2, 1e9, n_e) * b2.ms
        neuron_groups[name+'e'].tau_gi = truncated_normal_values(
            tau_gi/b2.ms, sigma_het, 0.4, 1e9, n_e) * b2.ms
        neuron_groups[name+'e'].refrac_e = truncated_normal_values(
            refrac_e/b2.ms, sigma_het, 1, 1e9, n_e) * b2.ms

        fig, ax = plt.subplots(2, 2, sharey=True)
        ax[0, 0].hist(neuron_groups[name+'e'].tau/b2.ms, bins=20)
        ax[0, 0].set_xlabel("τ [ms]")
        ax[0, 0].set_ylabel("Häufigkeit")
        ax[0, 0].set_title(
            "Verteilung der Zeitkonstante\n der Membranspannung")

        ax[0, 1].hist(neuron_groups[name+'e'].tau_ge/b2.ms, bins=20)
        ax[0, 1].set_xlabel("τ_ge [ms]")
        ax[0, 1].set_title(
            "Verteilung der Zeitkonstante\n der exz. Leitfähigkeit")

        ax[1, 0].hist(neuron_groups[name+'e'].tau_gi/b2.ms, bins=20)
        ax[1, 0].set_xlabel("τ_gi [ms]")
        ax[1, 0].set_ylabel("Häufigkeit")
        ax[1, 0].set_title(
            "Verteilung der Zeitkonstante\n der inh. Leitfähigkeit")

        ax[1, 1].hist(neuron_groups[name+'e'].refrac_e/b2.ms, bins=20)
        ax[1, 1].set_xlabel("Refraktärphase [ms]")
        ax[1, 1].set_title("Verteilung der Dauer\n der Refraktärphase")

        fig.suptitle("Neuronale Heterogenität")
        plt.tight_layout()
        plt.savefig(data_path+'plots/het_hist.png', dpi=600)
        plt.clf()

    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups[name+'e'].theta = np.load(
            weight_path + 'theta_' + name + ending + '.npy') * b2.volt
    else:
        neuron_groups[name+'e'].theta = np.ones((n_e)) * 20.0*b2.mV

    neuron_groups[name+'e'].rand_theta = min_rand_theta * \
        b2.mV + np.random.rand(n_e) * diff_rand_theta * b2.mV

    print(print_addon+'Creating recurrent connections...')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = get_matrix_from_file(
            weight_path + '../random/' + connName + ending + '.npy')
        model = 'w : 1'
        if p_dont_send_spike is not None and connName == 'AeAi':
            pre = 'g%s_post += w * int(rand() > p_dont_send_spike )' % conn_type[0]
        if p_dont_send_spike_inh is not None and connName == 'AiAe':
            pre = 'g%s_post += w * int(rand() > p_dont_send_spike_inh)' % conn_type[0]
        else:
            pre = 'g%s_post += w' % conn_type[0]
        post = ''
        if ee_STDP_on:
            if 'ee' in recurrent_conn_names:
                model += eqs_stdp_ee
                pre += '; ' + eqs_stdp_pre_ee
                post = eqs_stdp_post_ee
        connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                            model=model, on_pre=pre, on_post=post)
        connections[connName].connect(True)  # all-to-all connection
        connections[connName].w = weightMatrix[connections[connName].i,
                                               connections[connName].j]

    print(print_addon+'Creating monitors for', name)
    spike_counters[name +
                   'e'] = b2.SpikeMonitor(neuron_groups[name+'e'], record=False)

# ------------------------------------------------------------------------------
# create input population and connections from input populations
# ------------------------------------------------------------------------------
pop_values = [0, 0, 0]
for i, name in enumerate(input_population_names):
    input_groups[name+'e'] = b2.PoissonGroup(n_input, 0*Hz)

for name in input_connection_names:
    print(print_addon+'Creating connections between %s and %s...' %
          (name[0], name[1]))
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(
            weight_path + connName + ending + '.npy')
        model = 'w : 1'
        pre = 'g%s_post += w' % connType[0]
        post = ''
        if ee_STDP_on:
            print(print_addon+'Creating STDP for connection %s...' %
                  (name[0]+'e'+name[1]+'e'))
            model += eqs_stdp_ee
            pre += '; ' + eqs_stdp_pre_ee
            post = eqs_stdp_post_ee

        connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                            model=model, on_pre=pre, on_post=post)
        minDelay = delay[connType][0]
        maxDelay = delay[connType][1]
        deltaDelay = maxDelay - minDelay

        connections[connName].connect(True)  # all-to-all connection
        connections[connName].delay = 'minDelay + rand() * deltaDelay'
        if w_quant is not None:
            connections[connName].w = np.round(
                weightMatrix[connections[connName].i, connections[connName].j] * round_val_w) / round_val_w
        elif stoch_w_quant is not None:
            connections[connName].w = stoch_rounding_arr(
                weightMatrix[connections[connName].i, connections[connName].j], round_val_w)
        else:
            connections[connName].w = weightMatrix[connections[connName].i,
                                                   connections[connName].j]


# ------------------------------------------------------------------------------
# Printing summary
# ------------------------------------------------------------------------------

with open(data_path+'summary_%s.txt' % ('test'+"_"+test_label if test_mode else 'train'), 'w') as file:
    file.write(summary)

# ------------------------------------------------------------------------------
# Network operation to measure weight and voltage changes each timestep
# ------------------------------------------------------------------------------

if save_debug_info:
    weights = np.copy(connections['XeAe'].w)
    voltages = np.copy(neuron_groups['Ae'].v_)

    diff_w = []
    diff_v = []

    @b2.network_operation()
    def measure_changes(t):
        global weights, voltages, diff_w, diff_v, i_before

        if t > 0.51*b2.ms:
            diff_weights = np.abs(connections['XeAe'].w - weights)
            diff_voltages = np.abs(neuron_groups['Ae'].v_ - voltages)

            remove = np.logical_or(np.logical_or(np.abs(neuron_groups['Ae'].v - v_reset_e) > 1e-5 * b2.mV, np.abs(
                v_thresh_e - voltages*b2.volt) > 5 * b2.mV), diff_voltages > 1e-30)

            diff_weights = np.sort(diff_weights[diff_weights > 1e-30])
            diff_voltages = np.sort(diff_voltages[remove] * 1000)
            if len(diff_weights) > 0:
                diff_w.extend([quantile_sorted(diff_weights, p)
                              for p in np.arange(0., 1.+1e-5, 1e-1)])
            if len(diff_voltages) > 0:
                diff_v.extend([quantile_sorted(diff_voltages, p)
                              for p in np.arange(0., 1.+1e-5, 1e-1)])

        weights = np.copy(connections['XeAe'].w)
        voltages = np.copy(neuron_groups['Ae'].v_)


# ------------------------------------------------------------------------------
# run the simulation and set inputs
# ------------------------------------------------------------------------------

net = Network()
for obj_list in [neuron_groups, input_groups, connections, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

if save_debug_info:
    net.add(measure_changes)

previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
performance = np.zeros(int(num_examples / calculate_performance_interval))

for i, name in enumerate(input_population_names):
    input_groups[name+'e'].rates = 0 * Hz
net.run(0*second)

j = 0
last_time = time.time()
saved = False

if salt_pepper_alpha is not None and not test_mode:
    print("Plotting examples for salt and pepper noisy images...")
    plot_salt_and_pepper_examples(
        training['x'], 3, 5, salt_pepper_alpha, data_path)
if rectangle_noise_max is not None and not test_mode:
    print("Plotting examples for images with removed rectangle...")
    plot_remove_rectangle_examples(
        training['x'], 3, 5, rectangle_noise_min, rectangle_noise_max, data_path)

print('\n')
print("Starting simulation...")

while j < (int(num_examples)):
    if (not test_mode) or plasticity_during_testing:
        normalize_weights()
    if test_mode:
        mnist_arr = testing['x'][j % len(testing['x'])].reshape((n_input))
    else:
        mnist_arr = training['x'][j % len(training['x'])].reshape((n_input))
    if salt_pepper_alpha is not None:
        mnist_arr = salt_and_pepper(mnist_arr, salt_pepper_alpha)
    if rectangle_noise_max is not None:
        mnist_arr = remove_rectangle(
            mnist_arr, rectangle_noise_min, rectangle_noise_max)
    if is_mnist:
        spike_rates = mnist_arr / 8. * input_intensity
    else:
        spike_rates = mnist_arr / 24. * input_intensity
    input_groups['Xe'].rates = spike_rates * Hz
    net.run(single_example_time, report=None)

    current_spike_count = np.asarray(
        spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    if np.sum(current_spike_count) < 5 and input_intensity <= 6:
        input_intensity += 1
        if input_intensity == 6:
            print("j=%d: WARNING: input_intensity = 6! Number of spikes: %d" %
                  (j, np.sum(current_spike_count)))
        for i, name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)
    else:
        result_monitor[j, :] = current_spike_count
        if is_mnist:
            if test_mode:
                input_numbers[j] = testing['y'][j % len(testing['y'])][0]
            else:
                input_numbers[j] = training['y'][j % len(training['y'])][0]
        else:
            if test_mode:
                input_numbers[j] = testing['y'][j % len(testing['y'])]
            else:
                input_numbers[j] = training['y'][j % len(training['y'])]

        outputNumbers[j, :] = get_recognized_number_ranking(
            assignments, result_monitor[j, :])

        if j % print_progress_interval == print_progress_interval - 1:
            print_progress(j+1, int(num_examples), last_time,
                           print_progress_interval)
            last_time = time.time()
        for i, name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)

        if j % calculate_performance_interval == 0 and j > 0 and not test_mode:
            assignments = get_new_assignments(
                result_monitor[j-calculate_performance_interval: j], input_numbers[j-calculate_performance_interval: j])
            perf = get_current_performance(j)
            performance[int(j / calculate_performance_interval) - 1] = perf
            print("Performance for examples %d to %d: %3.3f" %
                  (j-calculate_performance_interval, j-1, 100*perf))
        if j % save_connections_interval == 0 and j > 0 and not test_mode:
            save_connections(str(j))
            save_theta(str(j))

        input_intensity = start_input_intensity
        j += 1

if j % calculate_performance_interval == 0 and not test_mode:
    assignments = get_new_assignments(
        result_monitor[j-calculate_performance_interval: j], input_numbers[j-calculate_performance_interval: j])
    perf = get_current_performance(j)
    performance[int(j / calculate_performance_interval) - 1] = perf
    print("Performance for examples %d to %d: %3.3f" %
          (j-calculate_performance_interval, j-1, 100*perf))


# ------------------------------------------------------------------------------
# save results
# ------------------------------------------------------------------------------
print(print_addon+'Saving results...')
if not test_mode:
    save_theta()
    save_connections()

print('Saving activity vectors and input numbers...')
suffix = 'test_'+test_label if test_mode else 'train'

np.save(data_path + 'activity/resultPopVecs_' + suffix, result_monitor)
np.save(data_path + 'activity/inputNumbers_' + suffix, input_numbers)

if not test_mode:
    print('Saving performance course...')
    np.save(data_path+'meta/performance', performance)

if save_debug_info:
    print("Saving debug plots...")
    print("Sorting arrays of length %d and %d" % (len(diff_w), len(diff_v)))
    diff_w = np.sort(diff_w)
    diff_v = np.sort(diff_v)

    print("Calculating quantiles...")
    x_vals_w = np.arange(start=0, stop=1+1e-6, step=1e-4)
    y_vals_w = np.array([quantile_sorted(diff_w, x) for x in x_vals_w])
    w_bits = [5, 6, 7, 8]

    x_vals_v = np.arange(start=0, stop=1+1e-6, step=1e-4)
    y_vals_v = np.array([quantile_sorted(diff_v, x) for x in x_vals_v])
    v_bits = [0, 1, 2, 3, 4]
    print("Plotting...")

    plt.rcParams['axes.axisbelow'] = True
    plt.style.use('bmh')

    accounted_w = [np.sum(diff_w >= pow(2, -val-1))/len(diff_w)
                   for val in w_bits]

    plt.plot(w_bits, accounted_w)
    plt.xlabel("Fractional quantization [Bit]")
    plt.ylabel("Proportion of weight updates\nincluded with this quantization")
    plt.title("Effect of synaptic weight quantization\non the weight update")
    plt.xticks(w_bits)
    plt.savefig(data_path+'plots/w_change_alt.png',
                dpi=600, bbox_inches="tight")
    plt.clf()

    accounted_v = [np.sum(diff_v >= pow(2, -val-1))/len(diff_w)
                   for val in v_bits]

    plt.plot(v_bits, accounted_v)
    plt.xlabel("Fractional quantization [Bit]")
    plt.ylabel("Proportion of voltage updates\nincluded with this quantization")
    plt.title("Effect of membrane voltage quantization\non the voltage update")
    plt.xticks(v_bits)
    plt.savefig(data_path+'plots/v_change_alt.png',
                dpi=600, bbox_inches="tight")
    plt.clf()

    plt.grid(True)
    plt.plot(x_vals_v, y_vals_v, label="measured voltage changes", zorder=0)
    for v_bit in v_bits:
        val = pow(2, -v_bit-1)
        plt.scatter([x_vals_v[np.argmin(np.abs(y_vals_v - val))]],
                    [val], label="%d Bits" % v_bit, marker='x', zorder=1)
    plt.yscale('log')
    plt.xlabel("p")
    plt.xticks(np.arange(start=0, stop=1+1e-3, step=1e-1))
    plt.ylabel("p-quantile [mV]")
    plt.title('Distribution of membrane voltage\nchanges per timestep')
    plt.legend()
    plt.savefig(data_path+'plots/v_change.png', dpi=600, bbox_inches="tight")
    plt.clf()

    plt.grid(True)
    plt.plot(x_vals_w, y_vals_w, label="measured weight changes", zorder=0)
    for w_bit in w_bits:
        val = pow(2, -w_bit-1)
        plt.scatter([x_vals_w[np.argmin(np.abs(y_vals_w - val))]],
                    [val], label="%d Bits" % w_bit, marker='x', zorder=1)
    plt.yscale('log')
    plt.xlabel("p")
    plt.ylabel("p-quantile [nS]")
    plt.xticks(np.arange(start=0, stop=1+1e-3, step=1e-1))
    plt.title('Distribution of weight changes\nper timestep')
    plt.legend()
    plt.savefig(data_path+'plots/w_change.png', dpi=600, bbox_inches="tight")
    plt.clf()

    np.savez(data_path+'meta/w_change_data.npz', w_bits=w_bits,
             accounted_w=accounted_w, x_vals_w=x_vals_w, y_vals_w=y_vals_w)
    np.savez(data_path+'meta/v_change_data.npz', v_bits=v_bits,
             accounted_v=accounted_v, x_vals_v=x_vals_v, y_vals_v=y_vals_v)

print(print_addon+'Finished!')
