import subprocess

additional_options = {
    'ref_6' : {},
    'ref_7' : {},
    'ref_8' : {},
    'ref_20000_1' : {'-train_size':20000},
    'ref_20000_2' : {'-train_size':20000},
    'ref_20000_3' : {'-train_size':20000},
    'v_quant_2': {'-membrane_voltage_quant': 2},
    'v_quant_3': {'-membrane_voltage_quant': 3},
    'v_quant_4': {'-membrane_voltage_quant': 4},
    'w_quant_4': {'-weight_quant': 4},
    'w_quant_5': {'-weight_quant': 5},
    'w_quant_6': {'-weight_quant': 6},
    'rand_threshold_1': {'-rand_threshold_max': 1},
    'rand_threshold_2': {'-rand_threshold_max': 2},
    'rand_threshold_3': {'-rand_threshold_max': 3},
    'rand_threshold_4': {'-rand_threshold_max': 4},
    'membrane_noise_1': {'-noise_membrane_voltage_max': 1},
    'membrane_noise_2': {'-noise_membrane_voltage_max': 2},
    'membrane_noise_3': {'-noise_membrane_voltage_max': 3},
    'membrane_noise_4': {'-noise_membrane_voltage_max': 4},
    'salt_and_pepper_10': {'-salt_and_pepper_alpha': 0.1},
    'salt_and_pepper_20': {'-salt_and_pepper_alpha': 0.2},
    'salt_and_pepper_30': {'-salt_and_pepper_alpha': 0.3},
    'salt_and_pepper_40': {'-salt_and_pepper_alpha': 0.4},
    'miss_spikes_10': {'-p_dont_send_spike': 0.1},
    'miss_spikes_20': {'-p_dont_send_spike': 0.2},
    'miss_spikes_30': {'-p_dont_send_spike': 0.3},
    'miss_spikes_40': {'-p_dont_send_spike': 0.4}
}


"""'ref_6' : {},
'ref_7' : {},
'ref_8' : {},
'ref_20000_1' : {'-train_size':20000},
'ref_20000_2' : {'-train_size':20000},
'ref_20000_3' : {'-train_size':20000},
'v_quant_2': {'-membrane_voltage_quant': 2},
'v_quant_3': {'-membrane_voltage_quant': 3},
'v_quant_4': {'-membrane_voltage_quant': 4},
'w_quant_4': {'-weight_quant': 4},
'w_quant_5': {'-weight_quant': 5},
'w_quant_6': {'-weight_quant': 6},
'rand_threshold_1': {'-rand_threshold_max': 1},
'rand_threshold_2': {'-rand_threshold_max': 2},
'rand_threshold_3': {'-rand_threshold_max': 3},
'rand_threshold_4': {'-rand_threshold_max': 4},
'membrane_noise_1': {'-noise_membrane_voltage_max': 1},
'membrane_noise_2': {'-noise_membrane_voltage_max': 2},
'membrane_noise_3': {'-noise_membrane_voltage_max': 3},
'membrane_noise_4': {'-noise_membrane_voltage_max': 4},
'salt_and_pepper_10': {'-salt_and_pepper_alpha': 0.1},
'salt_and_pepper_20': {'-salt_and_pepper_alpha': 0.2},
'salt_and_pepper_30': {'-salt_and_pepper_alpha': 0.3},
'salt_and_pepper_40': {'-salt_and_pepper_alpha': 0.4},
'miss_spikes_10': {'-p_dont_send_spike': 0.1},
'miss_spikes_20': {'-p_dont_send_spike': 0.2},
'miss_spikes_30': {'-p_dont_send_spike': 0.3},
'miss_spikes_40': {'-p_dont_send_spike': 0.4}"""


test_mode = True

def dict_to_args(dict):
    args = ''
    for key in dict:
        args += key + ' ' + str(dict[key])
    return args

for label in additional_options:
    nohup_file = './simulations/%s/simulation_%s.out' % (label, 'test' if test_mode else 'train')
    cmd_init = 'python init_directory_structure.py -label %s' % label
    cmd = 'python -u simulate.py -label %s -mode %s %s' % (label, 'test' if test_mode else 'train', dict_to_args(additional_options[label]))

    if not test_mode:
        print('Running '+cmd_init)
        finished_proc = subprocess.run(cmd_init.split())
    if test_mode or finished_proc.returncode == 0:
        out = open(nohup_file, 'w')
        print('Running '+cmd+'\n')
        subprocess.Popen(cmd.split(), stdout = out, stderr = out)
