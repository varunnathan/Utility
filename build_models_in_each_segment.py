# first segment based on the best smf model
# then, build models in each segment

import pandas as pd
import joblib
import numpy as np
import json
import os
from dstools.profile import calc_rmse
from sklearn.metrics import r2_score

# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
out_dir = project + 'segmented_models/'
inp_file = 'final_modelling_dataset_Oct172017_SC_segment{}.csv'
preprocess_out_file = 'final_modelling_dataset_Oct172017_SC_segment{}_preprocessed.pkl'
featsel_out_file = 'final_modelling_dataset_Oct172017_SC_segment{}_preprocessed_c60.pkl'
pipe_out_file = 'pipeline_modelling_dataset_Oct172017_SC_segment{}.pkl'
job = 'sc_preprocessed_c60_ols_segment{}'


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def preprocess(inp, out, pipe):
    cmd = 'kbglearn preprocess inf2nan exrsquaredimputer exoutscaler stdscaler --python 2 --target-col DV1 --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT  --sample-col sample --index-col ak_concat --datafile {} --output-fname {}; mv pipeline.pkl {}'.format(inp, out, pipe)
    os.system(cmd)


def feat_selection(inp, out, corr_coef=0.6):
    cmd = 'kbglearn featsel-cluster-contDV -c {} --python 2 --index-col ak_concat --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT --target-col DV1 --sample-col sample --datafile {}; mv data.pkl {}'.format(corr_coef, inp, out)
    os.system(cmd)


def grid(inp, jobname):
    cmd = 'kbglearn gridsearch -a ridge,lasso,gbr,rfr --python 2 --cv-folds 4 --random-state 54545454 --index-col ak_concat --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT --sample-col sample --target-col DV1 --datafile {} {} --queue'.format(inp, jobname)
    os.system(cmd)


if __name__ == '__main__':
    print 'segmented models are being developed...'
    print 'preprocess:'
    for item in range(1, 6):
        print item
        inp = raw_data+inp_file.format(item)
        out = out_dir+preprocess_out_file.format(item)
        pipe = out_dir+pipe_out_file.format(item)
        preprocess(inp, out, pipe)

    print 'feature selection:'
    for item in range(1, 6):
        print item
        inp = out_dir+preprocess_out_file.format(item)
        out = out_dir+featsel_out_file.format(item)
        feat_selection(inp, out)

    print 'gridsearch:'
    for item in range(1, 6):
        inp = out_dir+featsel_out_file.format(item)
        jobname = job.format(item)
        print 'jobname: %s' % (jobname)
        grid(inp=inp, jobname=jobname)
