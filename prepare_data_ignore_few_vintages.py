import pandas as pd
import joblib
import numpy as np
import json
import os

# GLOBALS
MI_flag = 'no'
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
preprocess_path = project + 'preprocess/'
feat_sel = project + 'feature_selection/'
grid_path = project + 'gridsearch/'
inp_file = 'final_modelling_dataset_Oct172017_SC_preprocessed.pkl'
out_file = 'final_modelling_dataset_Oct172017_SC_preprocessed_withoutMI.pkl'
featsel_out_file = 'final_modelling_dataset_Oct172017_SC_preprocessed_withoutMI_C60.pkl'
job = 'sc_preprocessed_c60_ols_withoutMI'
feats_all = 'CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT'
feats_without_MI = 'CR,EA,FICO,KH,LI,PAH,PP,RV,SC,YAH,YL'
if MI_flag == 'yes':
    feat_groups = ('CR', 'EA', 'FICO', 'KH', 'LI', 'MI', 'PAH', 'PP', 'RV',
                   'SC', 'YAH', 'YL', 'MKT')
    feats = feats_all
else:
    feat_groups = ('CR', 'EA', 'FICO', 'KH', 'LI', 'PAH', 'PP', 'RV',
                   'SC', 'YAH', 'YL')
    feats = feats_without_MI


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def filter_obs(data):
    df = data.copy()
    df['date'] = df['ak_concat'].apply(lambda x: x.split('_')[1])
    mask = (df['date'].isin(['2015-02-28', '2015-03-31', '2015-05-31',
                             '2015-06-30', '2016-05-31', '2015-07-31']) &
            (df['sample'] == 'dev'))
    df1 = df.loc[~mask, :]
    df1.reset_index(drop=True, inplace=True)
    return df1


def feat_sel_cluster(inp, out, feats, corr_coef=0.6):
    cmd = 'kbglearn featsel-cluster-contDV -c {} --python 2 --index-col ak_concat --feat-groups {} --target-col DV1 --sample-col sample --datafile {}; mv data.pkl {}'.format(corr_coef, feats, inp, out)
    os.system(cmd)


def grid(inp, feats, jobname):
    cmd = 'kbglearn gridsearch -a ridge,lasso,gbr,rfr --python 2 --cv-folds 4 --random-state 54545454 --index-col ak_concat --feat-groups {} --sample-col sample --target-col DV1 --datafile {} {} --queue'.format(feats, inp, jobname)
    os.system(cmd)


if __name__ == '__main__':
    print 'models without MI features...'
    df = read_datafile(preprocess_path+inp_file)
    # df1 = filter_obs(df)
    # df1.to_pickle(preprocess_path+out_file)

    # feature selection
    inp = preprocess_path+inp_file
    out = feat_sel + featsel_out_file
    feat_sel_cluster(inp, out, feats)

    # gridsearch
    inp = feat_sel + featsel_out_file
    grid(inp, feats, job)
