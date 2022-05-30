import subprocess

additional_options = {

}

N = 400
num_executions = 1
start = 1
test_mode = False
fashion_mnist = False

def dict_to_args(dict):
    args = ''
    for key in dict:
        args += key + ' ' + str(dict[key])+' '
    return args

for label in additional_options:
    args = dict_to_args(additional_options[label])
    temp = label.split("&&")
    label = temp[0]
    if len(temp) > 1:
        test_label = temp[1]
    else:
        test_label = "std"
    for i in range(start, start+num_executions):
        label_with_index = '%s_%d' % (label, i)
        nohup_file = '/mnt/data4tb/paessens/simulations/%s/simulation_%s.out' % (label_with_index, 'test_'+test_label if test_mode else 'train')
        cmd_init = 'python init_directory_structure.py -label %s -N %d' % (label_with_index, N)
        cmd = 'python -u simulate.py -label %s -mode %s %s -N %d' % (label_with_index, 'test' if test_mode else 'train', args, N)
        if fashion_mnist:
            cmd += ' -data fashion_mnist/'
        if test_mode:
            cmd += " -test_label "+test_label
        
        if not test_mode:
            print('Running '+cmd_init)
            finished_proc = subprocess.run(cmd_init.split())
        if test_mode or finished_proc.returncode == 0:
            out = open(nohup_file, 'w')
            print('Running '+cmd+'\n')
            subprocess.Popen(cmd.split(), stdout = out, stderr = out)
