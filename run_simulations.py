import subprocess

additional_options = {
    'new_ref_1':{}
}



num_executions = 10
test_mode = True
fashion_mnist = False

def dict_to_args(dict):
    args = ''
    for key in dict:
        args += key + ' ' + str(dict[key])+' '
    return args

for label in additional_options:
    args = dict_to_args(additional_options[label])
    for i in range(num_executions):
        if num_executions > 1:
            label_with_index = '%s_%d' % (label, i+1)
        else:
            label_with_index = label
        nohup_file = './simulations/%s/simulation_%s.out' % (label_with_index, 'test' if test_mode else 'train')
        cmd_init = 'python init_directory_structure.py -label %s' % label_with_index
        cmd = 'python -u simulate.py -label %s -mode %s %s' % (label_with_index, 'test' if test_mode else 'train', args)
        if fashion_mnist:
            cmd += ' -data fashion_mnist/'
        
        if not test_mode:
            print('Running '+cmd_init)
            finished_proc = subprocess.run(cmd_init.split())
        if test_mode or finished_proc.returncode == 0:
            out = open(nohup_file, 'w')
            print('Running '+cmd+'\n')
            subprocess.Popen(cmd.split(), stdout = out, stderr = out)
