#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2013/12/02';
__author__='Kensuke Mitsuzawa';

import re, subprocess, os, sys;
sys.path.append('/home/kensuke-mi/opt/liblinear-1.94/python/');
import liblinear, liblinearutil;
env='local';

if env=='pine':
    #for pine environment
    svmscale_exe='/home/kensuke-mi/opt/libsvm-3.17/svm-scale';
    liblinear_exe='/home/kensuke-mi/opt/liblinear-1.94/train';
    grid_py='/home/kensuke-mi/opt/libsvm-3.17/tools/grid.py';
elif env=='local':
    #for local environment
    svmscale_exe='~/opt/libsvm-3.17/svm-scale';
    liblinear_exe='~/opt/liblinear-1.94/train';
    grid_py='~/opt/libsvm-3.17/tools/grid.py';


def scalling_value(train_pathname, test_pathname):
    assert os.path.exists(train_pathname),"training file not found"
    file_name=os.path.basename(train_pathname);
    scaled_file = file_name + ".scale"
    model_file = file_name + ".model"
    range_file = file_name + ".range"
    file_name = os.path.split(test_pathname)[1]
    assert os.path.exists(test_pathname),"testing file not found"
    scaled_test_file = file_name + ".scale"
    predict_test_file = file_name + ".predict"
    cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_pathname, scaled_file)
    print('Scaling training data...')
    p = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE).communicate() 
    
    cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
    print('Scaling testing data...')
    p = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE).communicate() 
    
    return scaled_file, scaled_test_file;

def grid_search(target_file):
    #To check the C parameter for LIBLINEAR, use following command. "python grid.py -log2c -3,0,1 -log2g null -svmtrain ./train heart_scale"
    # from LIBLINEAR FAQ page: http://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html
    cmd = 'python {0} -log2c -3,0,1 -log2g null -svmtrain {1} {2}'.format(grid_py, liblinear_exe, target_file);
    print 'command for grid search is following:\n{}'.format(cmd);
    print('Cross validation...');
    f = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE).stdout
    line=''
    while True:
        print line;
        last_line=line
        line=f.readline()  
        if not line: break
    #outline format is [local] 0.0 92.7184 (best c=0.125, rate=94.0543)
    processed_line=re.sub(ur'\[local\]\s\.+\(best\sc=(.+),\srate=(.+)\)', ur'\1 \2', last_line);
    c,rate=map(float, processed_line.split());
    print('The result of grid search is Best c={0}, rate={1}'.format(c,rate));
    return c,rate;

def main(train_pathname, devset_pathname, scale):
    if scale==True:
        scaled_filepath, scaled_test_filepath=scalling_value(train_pathname, devset_pathname);
        c,rate=grid_search(scaled_filepath); 
        return c, rate, scaled_filepath, scaled_test_filepath;
    else:
        c,rate=grid_search(train_pathname); 
        return c, rate, None, None;

if __name__=='__main__':
    train_pathname=sys.argv[1];
    test_pathname=sys.argv[2];
    main(train_pathname, test_pathname, False);
