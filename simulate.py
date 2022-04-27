'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

# imports needed to parse arguments
import sys
import argparse
import os

######################################################
# ARGUMENT PARSING                                   #
######################################################

parser = argparse.ArgumentParser(description='''Simulation of SNN to classify MNIST images based on\
[https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full]. \
Original code can be found here: [https://github.com/peter-u-diehl/stdp-mnist].\
Migration of the original code to Brian2 and Python 3 can be found here: [https://github.com/sdpenguin/Brian2STDPMNIST].''')
parser.add_argument('-mode', dest='mode', type=str, help='Either test or training', required = True)
parser.add_argument('-label', dest='path', type=str, help='Label to save output files with', required = True)
parser.add_argument('-data', dest='datapath', type=str, help='Data path of MNIST dataset [./mnist/]', default = 'mnist/')
parser.add_argument('-epochs', dest='epochs', type=int, help='Number of epochs to train data with [1]', default = 1)
parser.add_argument('-train_size', dest='train_size', type=int, help='Number of inputs to train data each epoch with [60000]', default = 60000)
parser.add_argument('-test_size', dest='test_size', type=int, help='Number of inputs to test data with [10000]', default = 10000)
parser.add_argument('-plasticity', dest='plast_during_testing', help='Whether the network ist static\
or plastic during testing phase', action = 'store_true')
parser.add_argument('-rand_threshold_max', dest='rand_thresh_max', type = float, help='Maximal value of random threshold in mV [0mV]', default = 0.)
parser.add_argument('-rand_threshold_min', dest='rand_thresh_min', type = float, help='Minimal value of random threshold [0mV]', default = 0.)
parser.add_argument('-noise_membrane_voltage_max', dest='noise_membrane_voltage_max', type = float, help='Maximal value of random adjustment\
of membrane voltage per timestep for excitatory neurons [0mV]', default = 0.)
parser.add_argument('-noise_membrane_voltage_min', dest='noise_membrane_voltage_min', type = float, help='Minimal value of random adjustment\
of membrane voltage per timestep for excitatory neurons [0mV]', default = 0.)
parser.add_argument('-membrane_voltage_quant', dest='membrane_voltage_quant', type = int, help='Number of bits to quantify membane voltage of excitatory neurons [None]', default = None)
parser.add_argument('-weight_quant', dest='weight_quant', type = int, help='Number of bits to quantify weights [None]', default = None)
parser.add_argument('-salt_and_pepper_alpha', dest='salt_pepper_alpha', type = float, help='Propability that a pixel in the input image gets replaced by 0 or 255 [None]', default = None)
parser.add_argument('-rectangle_noise_min', dest='rectangle_noise_min', type = int, help='Minimal width and height of the rectangle that is removed from the input image [None]', default = None)
parser.add_argument('-rectangle_noise_max', dest='rectangle_noise_max', type = int, help='Maximal width and height of the rectangle that is removed from the input image [None]', default = None)
parser.add_argument('-p_dont_send_spike', dest='p_dont_send_spike', type = float, help='Propability that a spike in excitatory layer occurs without increasing\
the postsynaptic conductance in the inhibitory layer [None]', default = None)

args = parser.parse_args(sys.argv[1:])

test_mode = args.mode.upper() == 'TEST'
filename_label = args.path
MNIST_data_path = args.datapath
epochs = args.epochs
train_size = args.train_size
test_size = args.test_size
plasticity_during_testing = args.plast_during_testing
min_rand_theta = args.rand_thresh_min
diff_rand_theta = args.rand_thresh_max - min_rand_theta
noise_v_min = args.noise_membrane_voltage_min
noise_v_diff = args.noise_membrane_voltage_max - noise_v_min
v_quant = args.membrane_voltage_quant
w_quant = args.weight_quant
salt_pepper_alpha = args.salt_pepper_alpha
rectangle_noise_min = args.rectangle_noise_min
rectangle_noise_max = args.rectangle_noise_max
p_dont_send_spike = args.p_dont_send_spike

# if either max (or min) is None set max (min) to min (max)
# if both are None they stay None
if rectangle_noise_max is None:
    rectangle_noise_max = rectangle_noise_min
if rectangle_noise_min is None:
    rectangle_noise_min = rectangle_noise_max

print_addon = filename_label+": "

data_path = './simulations/'+filename_label +'/'
if not os.path.exists(data_path):
    raise Exception(print_addon+"Directory %s does not exist! Create it before running this script!" % data_path)

subfolder = ['plots', 'weights', 'activity', 'random', 'meta']
for subf in subfolder:
    if not os.path.exists(data_path+subf):
        raise Exception(print_addon+"Directory structure is incompletet! Directory %s is missing!" % (data_path+subf))

summary = '%s\nGeneral information:\n%s\n\ntest_mode = %s\nlabel = %s\nepochs = %d\ntrain size = %d\ntest size = %d\nplasticity during testing = %s\n' \
    % (60*'-',60*'-', test_mode, filename_label, epochs, train_size, test_size, plasticity_during_testing)

summary += '\n%s\nNoise information\n%s\n\n' % (60*'-',60*'-')
if min_rand_theta > 0 and diff_rand_theta > 0:
    summary += 'Minimal random threshold = %.5f\nMaximal random thersold = %.5f\n' % (min_rand_theta, min_rand_theta + diff_rand_theta)
if noise_v_min > 0 and noise_v_diff > 0:
    summary += 'Minimal random adjustment of the membrane voltage = %.5f\nMaximal random adjustment of the membrane voltage %.5f' % (noise_v_min, noise_v_min + noise_v_diff)
if v_quant is not None:
    summary += 'Number of bits to quantify membrane voltage: %d\n' % v_quant
if w_quant is not None:
    summary += 'Number of bits to quantify weights: %d\n' % w_quant
if salt_pepper_alpha is not None:
    summary += 'Propability of salt or pepper in input image: %.4f\n' % salt_pepper_alpha
if rectangle_noise_max is not None:
    summary += 'Minimal width of removed rectangle: %d\nMinimal width of removed rectangle: %d\n' % (rectangle_noise_min, rectangle_noise_max)
if p_dont_send_spike is not None:
    summary += 'Propability that a presynaptic spike does not lead to an increase of the postsynaptic conductance: %.4f' % (p_dont_send_spike)

# imports
import numpy as np
import time
from brian2 import *
import os
import brian2 as b2
from brian2tools import *
from tqdm import tqdm

# imports from own modules
from functions.data import get_labeled_data
from functions.input_noise import salt_and_pepper, remove_rectangle, plot_salt_and_pepper_examples, plot_remove_rectangle_examples

b2.prefs.codegen.target = 'cython'
b2.defaultclock.dt = 0.5 * b2.ms

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr

def save_connections(ending = ''):
    print(print_addon+'Saving connections...')
    for connName in save_conns:
        conn = connections[connName]
        connListSparse = list(zip(conn.i, conn.j, conn.w))
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    print(print_addon+'Saving theta...')
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

def normalize_weights():
    connName = 'XeAe'
    len_source = len(connections[connName].source)
    len_target = len(connections[connName].target)
    connection = np.zeros((len_source, len_target))
    connection[connections[connName].i, connections[connName].j] = connections[connName].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = weight['ee_input']/colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    if w_quant is not None:
        temp_conn = np.floor(temp_conn / round_val_w) * round_val_w
    connections[connName].w = temp_conn[connections[connName].i, connections[connName].j]

def get_current_performance(current_example_num):
    start_num = current_example_num - calculate_performance_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    return float(correct) / (end_num - start_num)

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments


#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------

start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print(print_addon+'Time needed to load training set: %.4fs' % (end - start))

start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print(print_addon+'Time needed to load test set: %.4fs' % (end - start))


#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------

np.random.seed(0)
if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = test_size
    ee_STDP_on = plasticity_during_testing
else:
    weight_path = data_path + 'random/'
    num_examples = train_size * epochs
    ee_STDP_on = True

record_spikes = True
calculate_performance_interval = 500
save_connections_interval = 1000
print_progress_interval = max(int(num_examples / 500), 1)

ending = ''
n_input = 784
n_e = 400
n_i = n_e
single_example_time =   0.35 * b2.second #
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
delay['ee_input'] = (0*b2.ms,10*b2.ms)
delay['ei_input'] = (0*b2.ms,5*b2.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 40*b2.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4
offset = 20.0*b2.mV

if w_quant is not None:
    round_val_w = wmax_ee/(2**w_quant)
    print("Quantifying weights to a bin of %.5f" % (round_val_w))
if v_quant is not None:
    round_val_v = (v_thresh_e - v_reset_e) / (2**v_quant)
    print("Quantifying membrane voltage to a bin of",round_val_v)


if test_mode and not plasticity_during_testing:
    if v_quant is not None:
        reset_e_str = 'x = v_reset_e; timer = 0*ms'
    else:
        reset_e_str = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b2.ms
    theta_plus_e = 0.05 * b2.mV
    if v_quant is not None:
        reset_e_str = 'x = v_reset_e; theta += theta_plus_e; timer = 0*ms'
    else:        
        reset_e_str = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

v_thresh_e_str = '(v > (rand_theta + theta - offset + v_thresh_e)) and (timer>refrac_e)'

reset_rand_theta = 'rand_theta = min_rand_theta*mV + rand()*diff_rand_theta*mV'
reset_e_str += '; '+  reset_rand_theta


v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'


if v_quant is not None:
    neuron_eqs_e = '''
            dx/dt = ((v_rest_e - x) + (I_synE+I_synI) / nS) / (100*ms) + (noise_v_min*mV + rand()*noise_v_diff * mV)/dt  : volt (unless refractory)
            v = floor(x / round_val_v) * round_val_v                : volt 
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-100.*mV-v)                         : amp
            dge/dt = -ge/(1.0*ms)                                   : 1
            dgi/dt = -gi/(2.0*ms)                                   : 1
            '''
else:
    neuron_eqs_e = '''
            dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms) + (noise_v_min*mV + rand()*noise_v_diff * mV)/dt  : volt (unless refractory)
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-100.*mV-v)                          : amp
            dge/dt = -ge/(1.0*ms)                                   : 1
            dgi/dt = -gi/(2.0*ms)                                  : 1
            '''

if test_mode and not plasticity_during_testing:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 1  : second'
neuron_eqs_e += '\n rand_theta :volt'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
if w_quant is None:
    eqs_stdp_pre_ee = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
    eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'
else:
    eqs_stdp_pre_ee = 'pre = 1.; w = floor(clip(w - nu_ee_pre * post1, 0, wmax_ee) / round_val_w) * round_val_w'
    eqs_stdp_post_ee = 'post2before = post2; w = floor(clip(w + nu_ee_post * pre * post2before, 0, wmax_ee) / round_val_w) * round_val_w; post1 = 1.; post2 = 1.'

b2.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((int(num_examples),n_e))

neuron_groups['e'] = b2.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e_str, refractory= refrac_e, reset= reset_e_str, method='euler')
neuron_groups['i'] = b2.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i_str, refractory= refrac_i, reset= v_reset_i_str, method='euler')

#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
for subgroup_n, name in enumerate(population_names):
    print(print_addon+'Creating neuron group %s...' % name)

    neuron_groups[name+'e'] = neuron_groups['e'][subgroup_n*n_e:(subgroup_n+1)*n_e]
    neuron_groups[name+'i'] = neuron_groups['i'][subgroup_n*n_i:(subgroup_n+1)*n_e]

    if v_quant is not None:
        neuron_groups[name+'e'].x = np.floor((v_rest_e - 40. * b2.mV) / round_val_v) * round_val_v
    else:
        neuron_groups[name+'e'].v = v_rest_e - 40. * b2.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b2.mV

    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy') * b2.volt
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b2.mV
        
    neuron_groups[name+'e'].rand_theta = min_rand_theta*b2.mV + np.random.rand(n_e) * diff_rand_theta * b2.mV

    print(print_addon+'Creating recurrent connections...')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + ending + '.npy')
        model = 'w : 1'
        if p_dont_send_spike is not None and connName == 'AeAi':
            pre = 'g%s_post += w * int(rand() > p_dont_send_spike )' % conn_type[0]
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
        connections[connName].connect(True) # all-to-all connection
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

    print(print_addon+'Creating monitors for', name)
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(neuron_groups[name+'e'])
    rate_monitors[name+'i'] = b2.PopulationRateMonitor(neuron_groups[name+'i'])
    spike_counters[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])

    if record_spikes:
        spike_monitors[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b2.SpikeMonitor(neuron_groups[name+'i'])


#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = b2.PoissonGroup(n_input, 0*Hz)
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(input_groups[name+'e'])

for name in input_connection_names:
    print(print_addon+'Creating connections between %s and %s...' %(name[0], name[1]))
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        model = 'w : 1'
        pre = 'g%s_post += w' % connType[0]
        post = ''
        if ee_STDP_on:
            print(print_addon+'Creating STDP for connection %s...' % (name[0]+'e'+name[1]+'e'))
            model += eqs_stdp_ee
            pre += '; ' + eqs_stdp_pre_ee
            post = eqs_stdp_post_ee

        connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        minDelay = delay[connType][0]
        maxDelay = delay[connType][1]
        deltaDelay = maxDelay - minDelay

        connections[connName].connect(True) # all-to-all connection
        connections[connName].delay = 'minDelay + rand() * deltaDelay'
        if w_quant is None:
            connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]
        else:
            connections[connName].w = np.floor(weightMatrix[connections[connName].i, connections[connName].j] / round_val_w) * round_val_w


#------------------------------------------------------------------------------
# Adding noise
#------------------------------------------------------------------------------

"""@b2.network_operation(when = 'end')
def debug(t):
    pass"""
    
#------------------------------------------------------------------------------
# Printing summary
#------------------------------------------------------------------------------

with open(data_path+'summary_%s.txt' % 'test' if test_mode else 'train', 'w') as file:
    file.write(summary)

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

net = Network()
for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
        spike_monitors, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

#net.add(debug)

previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

performance = np.zeros(int(num_examples / calculate_performance_interval))

for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rates = 0 * Hz
net.run(0*second)

j = 0
num_problem_saves = 0

if salt_pepper_alpha is not None:
    print("Plotting examples for salt and pepper noisy images...")
    plot_salt_and_pepper_examples(training['x'], 3,5, salt_pepper_alpha, filename_label)
if rectangle_noise_max is not None:
    print("Plotting examples for images with removed rectangle...")
    plot_remove_rectangle_examples(training['x'], 3,5, rectangle_noise_min, rectangle_noise_max, filename_label)

print('\n')
pbar = tqdm(total = int(num_examples), desc = print_addon+"Training progress")


while j < (int(num_examples)):
    if (not test_mode) or plasticity_during_testing:
        normalize_weights()
    if test_mode:
        mnist_arr = testing['x'][j%10000,:,:].reshape((n_input))
    else:
        mnist_arr = training['x'][j%60000,:,:].reshape((n_input))
    if salt_pepper_alpha is not None:
        mnist_arr = salt_and_pepper(mnist_arr, salt_pepper_alpha)
    if rectangle_noise_max is not None:
        mnist_arr = remove_rectangle(mnist_arr, rectangle_noise_min, rectangle_noise_max)
    spike_rates = mnist_arr / 8. *  input_intensity
    input_groups['Xe'].rates = spike_rates * Hz
    net.run(single_example_time, report=None)

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    if np.sum(current_spike_count) < 5 and input_intensity <= 10:
        input_intensity += 1
        if input_intensity == 10:
            print("j=%d: WARNING: input_intensity = 10! Number of spikes: %d" % (j, np.sum(current_spike_count)))
            if num_problem_saves < 15:
                np.save(data_path+"meta/no_spikes_input_%d" % j, spike_rates)
                save_connections("_problem_%d" % j)
                num_problem_saves += 1
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)
    else:
        result_monitor[j,:] = current_spike_count
        if test_mode:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j,:])

        if j % print_progress_interval == print_progress_interval - 1:
            pbar.update(print_progress_interval)
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)

        if j % calculate_performance_interval == 0 and j > 0 and not test_mode:
            assignments = get_new_assignments(result_monitor[j-calculate_performance_interval : j], input_numbers[j-calculate_performance_interval : j])
            perf = get_current_performance(j) 
            performance[int(j / calculate_performance_interval) - 1] = perf
            print("Performance for examples %d to %d: %3.3f" % (j-calculate_performance_interval, j-1, 100*perf))
        if j % save_connections_interval == 0 and j > 0 and not test_mode:
            save_connections(str(j))
            save_theta(str(j))

        input_intensity = start_input_intensity
        j += 1

if j % calculate_performance_interval == 0 and not test_mode:
    assignments = get_new_assignments(result_monitor[j-calculate_performance_interval : j], input_numbers[j-calculate_performance_interval : j])
    perf = get_current_performance(j) 
    performance[int(j / calculate_performance_interval) -1] = perf
    print("Performance for examples %d to %d: %3.3f" % (j-calculate_performance_interval, j-1, 100*perf))


#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print(print_addon+'Saving results...')
if not test_mode:
    save_theta()
    save_connections()

suffix = 'test' if test_mode else 'train'

np.save(data_path + 'activity/resultPopVecs_' + suffix, result_monitor)
np.save(data_path + 'activity/inputNumbers_' + suffix, input_numbers)

np.save(data_path+'meta/performance', performance)


#------------------------------------------------------------------------------
# plot results
#------------------------------------------------------------------------------
if rate_monitors:
    b2.figure(fig_num)
    fig_num += 1
    b2.figure(figsize = (10, 10))
    for i, name in enumerate(rate_monitors):
        b2.subplot(len(rate_monitors), 1, 1+i)
        b2.plot(rate_monitors[name].t/b2.second, rate_monitors[name].rate, '.')
        b2.title('Rates of population ' + name)
        np.save(data_path+'meta/rate_monitor_'+name+'_times', rate_monitors[name].t/b2.second)
        np.save(data_path+'meta/rate_monitor_'+name+'_rates', rate_monitors[name].rate)
    b2.tight_layout()
    b2.savefig(data_path+"plots/rate_monitors.png", dpi = 600)

if spike_monitors:
    b2.figure(fig_num)
    fig_num += 1
    b2.figure(figsize = (10, 10))
    for i, name in enumerate(spike_monitors):
        b2.subplot(len(spike_monitors), 1, 1+i)
        b2.plot(spike_monitors[name].t/b2.ms, spike_monitors[name].i, '.')
        b2.title('Spikes of population ' + name)
        np.save(data_path+'meta/spike_monitor_'+name+'_times', spike_monitors[name].t/b2.ms)
        np.save(data_path+'meta/spike_monitor_'+name+'_ids', spike_monitors[name].i)
    b2.tight_layout()
    b2.savefig(data_path+"plots/spike_monitors.png", dpi = 600)

if spike_counters:
    b2.figure(fig_num)
    fig_num += 1
    b2.plot(spike_monitors['Ae'].count[:])
    b2.title('Spike count of population Ae')
    b2.savefig(data_path+"plots/spike_counter.png", dpi = 600)

print(print_addon+'Finished!')